"""Crypto rails: agents pay in a stablecoin (default USDC on Base) to one
platform treasury address; the platform keeps its fee and pays sellers out.

  Link wallet   An agent proves it controls an address by signing a one-time
                challenge (EIP-191 personal_sign). Only then are transfers from
                that address credited to it, so nobody can claim someone
                else's deposit.
  Deposits      A watcher polls the chain for token Transfer events into the
                treasury. After N confirmations each one is recorded once
                (key: chain:tx:log_index). From a linked wallet it is credited
                (house:crypto_deposits -> agent available); from an unknown
                sender it is held as "unattributed" and credited automatically
                if that sender links later.
  Payouts       An agent withdraws to one of its own linked wallets. The
                ledger debits it immediately (-> house:crypto_payable). The
                server holds no private key: the operator sends the transfer
                from the treasury and submits the tx hash, and the payout is
                marked paid only after the transfer is verified on-chain
                (right token, from treasury, to that address, exact amount,
                enough confirmations). Cancelling refunds the agent.
  Commission    The clearing fee accrues in house:fees; the USDC itself stays
                in the treasury. GET /v1/admin/crypto/solvency compares the
                on-chain balance with everything owed.
"""

import hmac
import logging
import re
import secrets
import time
from decimal import Decimal

import httpx
from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from . import ledger
from .auth import AuthContext, authenticate
from .config import settings
from .db import SessionLocal, utcnow
from .models import Agent, ChainCursor, CryptoDeposit, CryptoPayout, LinkedWallet, TradeLedger
from .units import NANOS_PER_USD, fmt_usd

log = logging.getLogger("aether.crypto")

router = APIRouter(tags=["crypto"])

TRANSFER_TOPIC = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"  # Transfer(address,address,uint256)
BALANCE_OF = "0x70a08231"
ADDRESS_RE = re.compile(r"^0x[0-9a-fA-F]{40}$")


def norm_address(value: str) -> str:
    if not ADDRESS_RE.match(value or ""):
        raise HTTPException(422, detail="not a 0x-prefixed 20-byte address")
    return value.lower()


def topic_for(address: str) -> str:
    return "0x" + "0" * 24 + address[2:].lower()


def address_from_topic(topic: str) -> str:
    return "0x" + topic[-40:].lower()


def nanos_per_unit() -> int:
    return NANOS_PER_USD // 10**settings.crypto_token_decimals


def units_to_nanos(units: int) -> int:
    return units * nanos_per_unit()


def fmt_token(units: int) -> str:
    return f"{Decimal(units) / 10**settings.crypto_token_decimals} {settings.crypto_token_symbol}"


class ChainError(Exception):
    pass


class EvmClient:
    """Minimal JSON-RPC client for the few calls the rails need."""

    def __init__(self, http: httpx.AsyncClient, url: str):
        self.http = http
        self.url = url
        self._id = 0

    @classmethod
    def create(cls) -> "EvmClient | None":
        if not settings.crypto_treasury_address:
            return None
        if not ADDRESS_RE.match(settings.crypto_treasury_address):
            raise RuntimeError("AETHER_CRYPTO_TREASURY_ADDRESS is not a valid 0x address")
        if settings.crypto_token_decimals > 9:
            raise RuntimeError("tokens with more than 9 decimals are not supported by the nano-USD ledger")
        return cls(httpx.AsyncClient(timeout=httpx.Timeout(15.0)), settings.crypto_rpc_url)

    async def call(self, method: str, params: list):
        self._id += 1
        try:
            resp = await self.http.post(self.url, json={"jsonrpc": "2.0", "id": self._id, "method": method, "params": params})
            body = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise ChainError(f"{method}: {exc}") from None
        if "error" in body:
            raise ChainError(f"{method}: {body['error']}")
        return body.get("result")

    async def block_number(self) -> int:
        return int(await self.call("eth_blockNumber", []), 16)

    async def transfers_to(self, token: str, to_address: str, from_block: int, to_block: int) -> list[dict]:
        return await self.call(
            "eth_getLogs",
            [
                {
                    "fromBlock": hex(from_block),
                    "toBlock": hex(to_block),
                    "address": token,
                    "topics": [TRANSFER_TOPIC, None, topic_for(to_address)],
                }
            ],
        )

    async def receipt(self, tx_hash: str) -> dict | None:
        return await self.call("eth_getTransactionReceipt", [tx_hash])

    async def token_balance(self, token: str, holder: str) -> int:
        data = BALANCE_OF + "0" * 24 + holder[2:].lower()
        return int(await self.call("eth_call", [{"to": token, "data": data}, "latest"]) or "0x0", 16)

    async def aclose(self) -> None:
        await self.http.aclose()


def _chain(request: Request) -> EvmClient:
    client = getattr(request.app.state, "chain", None)
    if client is None:
        raise HTTPException(503, detail="crypto payments are not configured on this clearinghouse")
    return client


def require_admin(x_admin_token: str | None) -> None:
    """Money-moving admin actions always need the admin token, even in sandbox mode."""
    if not settings.admin_token or x_admin_token is None or not hmac.compare_digest(x_admin_token, settings.admin_token):
        raise HTTPException(403, detail="admin token required")


# ------------------------------------------------------------------- deposits


async def record_deposit(deposit_id: str, tx_hash: str, from_address: str, units: int, block: int) -> str | None:
    """Idempotently record one Transfer into the treasury; credit it if the
    sender is a linked wallet. Returns the new status, or None if already seen."""
    try:
        async with SessionLocal() as session, session.begin():
            if await session.get(CryptoDeposit, deposit_id) is not None:
                return None
            wallet = await session.get(LinkedWallet, from_address)
            deposit = CryptoDeposit(
                id=deposit_id,
                tx_hash=tx_hash,
                from_address=from_address,
                amount_units=units,
                block_number=block,
                agent_id=wallet.agent_id if wallet else None,
                status="credited" if wallet else "unattributed",
            )
            session.add(deposit)
            await session.flush()
            if wallet:
                _credit(session, await ledger.lock_agents(session, [wallet.agent_id]), deposit)
    except IntegrityError:
        return None  # a concurrent watcher recorded it first
    log.info("DEPOSIT %s %s from %s tx=%s", deposit.status, fmt_token(units), from_address, tx_hash)
    return deposit.status


def _credit(session, agents, deposit: CryptoDeposit) -> None:
    nanos = units_to_nanos(deposit.amount_units)
    ledger.post(
        session,
        agents,
        [ledger.house_leg(ledger.HOUSE_CRYPTO_IN, -nanos), ledger.agent_leg(deposit.agent_id, ledger.AVAILABLE, nanos)],
        kind="deposit",
        ref_type="crypto",
        ref_id=deposit.id,
        memo=f"{fmt_token(deposit.amount_units)} tx {deposit.tx_hash}",
    )


class DepositWatcher:
    def __init__(self, chain: EvmClient):
        self.chain = chain
        self._next_run = 0.0

    async def maybe_poll(self) -> None:
        if time.monotonic() < self._next_run:
            return
        self._next_run = time.monotonic() + settings.crypto_poll_interval_s
        await self.poll()

    async def poll(self, max_batches: int = 20) -> int:
        treasury = settings.crypto_treasury_address.lower()
        token = settings.crypto_token_address.lower()
        safe = await self.chain.block_number() - settings.crypto_confirmations
        async with SessionLocal() as session, session.begin():
            cursor = await session.get(ChainCursor, settings.crypto_chain_id)
            if cursor is None:
                start = settings.crypto_start_block if settings.crypto_start_block is not None else safe
                cursor = ChainCursor(chain_id=settings.crypto_chain_id, last_block=start - 1)
                session.add(cursor)
            last = cursor.last_block
        seen = 0
        for _ in range(max_batches):
            frm = last + 1
            if frm > safe:
                break
            to = min(safe, frm + settings.crypto_max_block_range - 1)
            for entry in await self.chain.transfers_to(token, treasury, frm, to):
                if entry.get("removed"):
                    continue
                if entry["address"].lower() != token or entry["topics"][0] != TRANSFER_TOPIC:
                    continue
                deposit_id = f"{settings.crypto_chain_id}:{entry['transactionHash'].lower()}:{int(entry['logIndex'], 16)}"
                status = await record_deposit(
                    deposit_id,
                    entry["transactionHash"].lower(),
                    address_from_topic(entry["topics"][1]),
                    int(entry["data"], 16),
                    int(entry["blockNumber"], 16),
                )
                seen += status is not None
            async with SessionLocal() as session, session.begin():
                cursor = (
                    await session.execute(
                        select(ChainCursor).where(ChainCursor.chain_id == settings.crypto_chain_id).with_for_update()
                    )
                ).scalar_one()
                cursor.last_block = max(cursor.last_block, to)
            last = to
        return seen


# -------------------------------------------------------------- agent routes


class LinkWalletIn(BaseModel):
    address: str
    nonce: str
    signature: str


class CryptoWithdrawalIn(BaseModel):
    amount_usd: Decimal = Field(gt=0, decimal_places=2)
    to_address: str


def _link_message(agent_id: str, address: str, nonce: str) -> str:
    return f"Aether wallet link\nagent: {agent_id}\naddress: {address}\nchain: {settings.crypto_chain_id}\nnonce: {nonce}"


def _nonce_key(agent_id: str, nonce: str) -> str:
    return f"{settings.redis_prefix}:cryptolink:{agent_id}:{nonce}"


@router.get("/v1/billing/crypto")
async def deposit_instructions(request: Request, ctx: AuthContext = Depends(authenticate)):
    _chain(request)
    async with SessionLocal() as session:
        wallets = (
            (await session.execute(select(LinkedWallet.address).where(LinkedWallet.agent_id == ctx.agent_id))).scalars().all()
        )
    return {
        "chain": settings.crypto_chain_name,
        "chain_id": settings.crypto_chain_id,
        "token": settings.crypto_token_symbol,
        "token_contract": settings.crypto_token_address,
        "treasury_address": settings.crypto_treasury_address,
        "confirmations": settings.crypto_confirmations,
        "linked_wallets": list(wallets),
        "instructions": (
            f"1) Link a wallet: GET /v1/billing/crypto/link-challenge?address=<yours>, sign the message with it, "
            f"POST /v1/billing/crypto/wallets. 2) Send {settings.crypto_token_symbol} on {settings.crypto_chain_name} "
            "from that wallet to treasury_address. Other tokens or chains are not credited."
        ),
    }


@router.get("/v1/billing/crypto/link-challenge")
async def link_challenge(address: str, request: Request, ctx: AuthContext = Depends(authenticate)):
    _chain(request)
    addr = norm_address(address)
    nonce = secrets.token_hex(16)
    await request.app.state.redis.set(_nonce_key(ctx.agent_id, nonce), addr, ex=600)
    return {"message": _link_message(ctx.agent_id, addr, nonce), "nonce": nonce, "expires_in": 600}


@router.post("/v1/billing/crypto/wallets", status_code=201)
async def link_wallet(body: LinkWalletIn, request: Request, ctx: AuthContext = Depends(authenticate)):
    _chain(request)
    addr = norm_address(body.address)
    expected = await request.app.state.redis.getdel(_nonce_key(ctx.agent_id, body.nonce))
    if expected != addr:
        raise HTTPException(400, detail="unknown, expired or already used challenge")
    from eth_account import Account  # imported lazily: only needed when crypto is on
    from eth_account.messages import encode_defunct

    try:
        signer = Account.recover_message(
            encode_defunct(text=_link_message(ctx.agent_id, addr, body.nonce)), signature=body.signature
        )
    except Exception:
        raise HTTPException(400, detail="malformed signature") from None
    if signer.lower() != addr:
        raise HTTPException(400, detail="signature was not made by this address")

    credited = 0
    try:
        async with SessionLocal() as session, session.begin():
            existing = await session.get(LinkedWallet, addr)
            if existing is not None and existing.agent_id != ctx.agent_id:
                raise HTTPException(409, detail="address is linked to another agent")
            if existing is None:
                session.add(LinkedWallet(address=addr, agent_id=ctx.agent_id))
                await session.flush()
            # Deposits that arrived before the wallet was linked are credited now.
            pending = (
                (
                    await session.execute(
                        select(CryptoDeposit)
                        .where(CryptoDeposit.from_address == addr, CryptoDeposit.status == "unattributed")
                        .with_for_update()
                    )
                )
                .scalars()
                .all()
            )
            if pending:
                agents = await ledger.lock_agents(session, [ctx.agent_id])
                for deposit in pending:
                    deposit.agent_id, deposit.status = ctx.agent_id, "credited"
                    _credit(session, agents, deposit)
                    credited += units_to_nanos(deposit.amount_units)
    except IntegrityError:
        raise HTTPException(409, detail="address is linked to another agent") from None
    return {"address": addr, "linked": True, "credited_now_usd": fmt_usd(credited)}


@router.get("/v1/billing/crypto/deposits")
async def my_deposits(ctx: AuthContext = Depends(authenticate)):
    async with SessionLocal() as session:
        rows = (
            (
                await session.execute(
                    select(CryptoDeposit)
                    .where(CryptoDeposit.agent_id == ctx.agent_id)
                    .order_by(CryptoDeposit.block_number.desc())
                )
            )
            .scalars()
            .all()
        )
    return [
        {"id": d.id, "tx_hash": d.tx_hash, "amount": fmt_token(d.amount_units), "block": d.block_number, "status": d.status}
        for d in rows
    ]


@router.post("/v1/billing/crypto/withdrawals", status_code=201)
async def request_withdrawal(body: CryptoWithdrawalIn, request: Request, ctx: AuthContext = Depends(authenticate)):
    """Withdraw available balance to one of the agent's own linked wallets."""
    _chain(request)
    if body.amount_usd < settings.crypto_min_withdrawal_usd:
        raise HTTPException(422, detail=f"minimum withdrawal is ${settings.crypto_min_withdrawal_usd}")
    to = norm_address(body.to_address)
    units = int(body.amount_usd * 10**settings.crypto_token_decimals)
    nanos = units_to_nanos(units)
    async with SessionLocal() as session, session.begin():
        wallet = await session.get(LinkedWallet, to)
        if wallet is None or wallet.agent_id != ctx.agent_id:
            raise HTTPException(409, detail="payouts go only to a wallet this agent has linked")
        agents = await ledger.lock_agents(session, [ctx.agent_id])
        payout = CryptoPayout(agent_id=ctx.agent_id, to_address=to, amount_units=units, status="pending")
        session.add(payout)
        await session.flush()
        try:
            ledger.post(
                session,
                agents,
                [ledger.agent_leg(ctx.agent_id, ledger.AVAILABLE, -nanos), ledger.house_leg(ledger.HOUSE_CRYPTO_PAYABLE, nanos)],
                kind="withdrawal",
                ref_type="crypto_payout",
                ref_id=payout.id,
            )
        except ledger.InsufficientFunds:
            raise HTTPException(402, detail="insufficient available balance") from None
    log.info("PAYOUT requested %s agent=%s to=%s id=%s", fmt_token(units), ctx.agent_id, to, payout.id)
    return {"payout_id": payout.id, "status": "pending", "amount": fmt_token(units), "to_address": to}


@router.get("/v1/billing/crypto/withdrawals")
async def my_withdrawals(ctx: AuthContext = Depends(authenticate)):
    async with SessionLocal() as session:
        rows = (
            (
                await session.execute(
                    select(CryptoPayout).where(CryptoPayout.agent_id == ctx.agent_id).order_by(CryptoPayout.created_at.desc())
                )
            )
            .scalars()
            .all()
        )
    return [_payout_out(p) for p in rows]


def _payout_out(p: CryptoPayout) -> dict:
    return {
        "payout_id": p.id,
        "agent_id": p.agent_id,
        "to_address": p.to_address,
        "amount": fmt_token(p.amount_units),
        "amount_units": p.amount_units,
        "status": p.status,
        "tx_hash": p.tx_hash,
        "created_at": p.created_at,
        "paid_at": p.paid_at,
    }


# -------------------------------------------------------------- admin routes


class MarkPaidIn(BaseModel):
    tx_hash: str = Field(pattern=r"^0x[0-9a-fA-F]{64}$")


@router.get("/v1/admin/crypto/payouts")
async def admin_payouts(status: str = "pending", x_admin_token: str | None = Header(None)):
    require_admin(x_admin_token)
    async with SessionLocal() as session:
        rows = (
            (await session.execute(select(CryptoPayout).where(CryptoPayout.status == status).order_by(CryptoPayout.created_at)))
            .scalars()
            .all()
        )
    return {
        "token": settings.crypto_token_symbol,
        "token_contract": settings.crypto_token_address,
        "payouts": [_payout_out(p) for p in rows],
    }


async def _verify_transfer(chain: EvmClient, tx_hash: str, to: str, units: int) -> None:
    receipt = await chain.receipt(tx_hash)
    if not receipt:
        raise HTTPException(409, detail="transaction not found (not mined yet?)")
    if receipt.get("status") != "0x1":
        raise HTTPException(409, detail="transaction failed on-chain")
    confirmations = await chain.block_number() - int(receipt["blockNumber"], 16) + 1
    if confirmations < settings.crypto_confirmations:
        raise HTTPException(409, detail=f"only {confirmations}/{settings.crypto_confirmations} confirmations; retry shortly")
    treasury, token = settings.crypto_treasury_address.lower(), settings.crypto_token_address.lower()
    for entry in receipt.get("logs", []):
        topics = entry.get("topics", [])
        if (
            entry.get("address", "").lower() == token
            and len(topics) == 3
            and topics[0] == TRANSFER_TOPIC
            and address_from_topic(topics[1]) == treasury
            and address_from_topic(topics[2]) == to
            and int(entry["data"], 16) == units
        ):
            return
    raise HTTPException(
        409,
        detail=f"no {settings.crypto_token_symbol} transfer of {fmt_token(units)} from the treasury to {to} in that transaction",
    )


@router.post("/v1/admin/crypto/payouts/{payout_id}/paid")
async def admin_mark_paid(payout_id: str, body: MarkPaidIn, request: Request, x_admin_token: str | None = Header(None)):
    require_admin(x_admin_token)
    chain = _chain(request)
    async with SessionLocal() as session:
        payout = await session.get(CryptoPayout, payout_id)
    if payout is None:
        raise HTTPException(404, detail="payout not found")
    if payout.status != "pending":
        raise HTTPException(409, detail=f"payout is already {payout.status}")
    tx_hash = body.tx_hash.lower()
    await _verify_transfer(chain, tx_hash, payout.to_address, payout.amount_units)
    nanos = units_to_nanos(payout.amount_units)
    try:
        async with SessionLocal() as session, session.begin():
            payout = (
                await session.execute(select(CryptoPayout).where(CryptoPayout.id == payout_id).with_for_update())
            ).scalar_one()
            if payout.status != "pending":
                raise HTTPException(409, detail=f"payout is already {payout.status}")
            payout.status, payout.tx_hash, payout.paid_at = "paid", tx_hash, utcnow()
            await session.flush()
            ledger.post(
                session,
                {},
                [ledger.house_leg(ledger.HOUSE_CRYPTO_PAYABLE, -nanos), ledger.house_leg(ledger.HOUSE_CRYPTO_OUT, nanos)],
                kind="payout_sent",
                ref_type="crypto_payout",
                ref_id=payout_id,
                memo=f"tx {tx_hash}",
            )
    except IntegrityError:
        raise HTTPException(409, detail="that transaction is already recorded for another payout") from None
    log.info("PAYOUT paid %s id=%s tx=%s", fmt_token(payout.amount_units), payout_id, tx_hash)
    return _payout_out(payout)


@router.post("/v1/admin/crypto/payouts/{payout_id}/cancel")
async def admin_cancel(payout_id: str, x_admin_token: str | None = Header(None)):
    require_admin(x_admin_token)
    async with SessionLocal() as session, session.begin():
        payout = (
            await session.execute(select(CryptoPayout).where(CryptoPayout.id == payout_id).with_for_update())
        ).scalar_one_or_none()
        if payout is None:
            raise HTTPException(404, detail="payout not found")
        if payout.status != "pending":
            raise HTTPException(409, detail=f"payout is already {payout.status}")
        agents = await ledger.lock_agents(session, [payout.agent_id])
        nanos = units_to_nanos(payout.amount_units)
        ledger.post(
            session,
            agents,
            [ledger.house_leg(ledger.HOUSE_CRYPTO_PAYABLE, -nanos), ledger.agent_leg(payout.agent_id, ledger.AVAILABLE, nanos)],
            kind="withdrawal_reversal",
            ref_type="crypto_payout",
            ref_id=payout_id,
            memo="cancelled by operator",
        )
        payout.status = "cancelled"
    return _payout_out(payout)


@router.get("/v1/admin/crypto/solvency")
async def admin_solvency(request: Request, x_admin_token: str | None = Header(None)):
    """Is the treasury's on-chain balance enough to cover every obligation?"""
    require_admin(x_admin_token)
    chain = _chain(request)
    onchain_units = await chain.token_balance(settings.crypto_token_address, settings.crypto_treasury_address)
    async with SessionLocal() as session:
        balances = (
            await session.execute(select(func.coalesce(func.sum(Agent.balance_available_nanos + Agent.balance_escrow_nanos), 0)))
        ).scalar_one()
        by_account = dict(
            (
                await session.execute(
                    select(TradeLedger.account, func.sum(TradeLedger.amount_nanos))
                    .where(
                        TradeLedger.account.in_(
                            [ledger.HOUSE_FEES, ledger.HOUSE_CRYPTO_PAYABLE, ledger.HOUSE_CRYPTO_IN, ledger.HOUSE_CRYPTO_OUT]
                        )
                    )
                    .group_by(TradeLedger.account)
                )
            ).all()
        )
        unattributed = (
            await session.execute(
                select(func.coalesce(func.sum(CryptoDeposit.amount_units), 0)).where(CryptoDeposit.status == "unattributed")
            )
        ).scalar_one()
    onchain = units_to_nanos(onchain_units)
    owed_agents = int(balances)
    payable = int(by_account.get(ledger.HOUSE_CRYPTO_PAYABLE, 0))
    fees = int(by_account.get(ledger.HOUSE_FEES, 0))
    obligations = owed_agents + payable
    return {
        "treasury_address": settings.crypto_treasury_address,
        "onchain_balance": fmt_token(onchain_units),
        "owed_to_agents_usd": fmt_usd(owed_agents),
        "pending_payouts_usd": fmt_usd(payable),
        "fees_earned_usd": fmt_usd(fees),
        "unattributed_deposits": fmt_token(int(unattributed)),
        "deposited_total_usd": fmt_usd(-int(by_account.get(ledger.HOUSE_CRYPTO_IN, 0))),
        "paid_out_total_usd": fmt_usd(int(by_account.get(ledger.HOUSE_CRYPTO_OUT, 0))),
        "surplus_usd": fmt_usd(onchain - obligations),
        "solvent": onchain >= obligations,
    }


async def run_watcher_pass(app) -> None:
    watcher = getattr(app.state, "deposit_watcher", None)
    if watcher is not None:
        try:
            await watcher.maybe_poll()
        except ChainError as exc:
            log.warning("deposit watcher: %s", exc)


__all__ = ["DepositWatcher", "EvmClient", "record_deposit", "router", "run_watcher_pass"]
