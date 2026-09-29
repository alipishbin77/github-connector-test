"""Crypto rails: agents pay in USDC on any supported EVM network to ONE
platform treasury address; the platform keeps its fee and pays sellers out.

  Networks      AETHER_CRYPTO_NETWORKS="ethereum,base,arbitrum,optimism,polygon"
                (presets for native USDC), or a JSON list for custom tokens.
                The same treasury address receives on every network, so it
                must be a plain wallet (EOA) that Ali controls on all of them.
  Link wallet   An agent proves it controls an address by signing a one-time
                challenge (EIP-191). Linking once covers every network.
                Only then are transfers from that address credited to it.
  Deposits      One watcher per network polls token Transfer events into the
                treasury and records each once (key chain:tx:log_index) after
                that network's confirmations. From a linked wallet it is
                credited; from an unknown sender it is held "unattributed" and
                credited automatically if that sender links later.
  Payouts       Withdrawals go only to the agent's own linked wallet, on the
                network it picks (default: cheapest configured). The ledger
                debits it immediately. The server holds no private key: the
                operator sends from the treasury and the payout is marked paid
                only after the transfer is verified on that network.
  Commission    The fee accrues in house:fees; the USDC stays in the treasury.
                GET /v1/admin/crypto/solvency sums on-chain balances across
                networks and compares them with everything owed.
"""

import hmac
import json
import logging
import re
import secrets
import time
from dataclasses import dataclass, replace
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


@dataclass(frozen=True)
class Network:
    key: str
    name: str
    chain_id: int
    rpc_url: str
    token_address: str
    token_symbol: str = "USDC"
    decimals: int = 6
    confirmations: int = 12
    max_block_range: int = 500

    @property
    def nanos_per_unit(self) -> int:
        return NANOS_PER_USD // 10**self.decimals

    def to_nanos(self, units: int) -> int:
        return units * self.nanos_per_unit

    def fmt(self, units: int) -> str:
        return f"{Decimal(units) / 10**self.decimals} {self.token_symbol} on {self.name}"

    def public(self) -> dict:
        return {
            "network": self.key,
            "name": self.name,
            "chain_id": self.chain_id,
            "token": self.token_symbol,
            "token_contract": self.token_address,
            "decimals": self.decimals,
            "confirmations": self.confirmations,
        }


# Native (Circle-issued) USDC. Bridged variants such as USDC.e are NOT credited.
PRESETS: dict[str, Network] = {
    "ethereum": Network("ethereum", "Ethereum", 1, "https://ethereum-rpc.publicnode.com",
                        "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48", confirmations=12, max_block_range=500),
    "base": Network("base", "Base", 8453, "https://base-rpc.publicnode.com",
                    "0x833589fcd6edb6e08f4c7c32d4f71b54bda02913", confirmations=10, max_block_range=1000),
    "arbitrum": Network("arbitrum", "Arbitrum One", 42161, "https://arbitrum-one-rpc.publicnode.com",
                        "0xaf88d065e77c8cc2239327c5edb3a432268e5831", confirmations=240, max_block_range=5000),
    "optimism": Network("optimism", "OP Mainnet", 10, "https://optimism-rpc.publicnode.com",
                        "0x0b2c639c533813f4aa9d7837caf62653d097ff85", confirmations=10, max_block_range=1000),
    "polygon": Network("polygon", "Polygon PoS", 137, "https://polygon-bor-rpc.publicnode.com",
                       "0x3c499c542cef5e3811e1192ce70d8cc03d5c3359", confirmations=128, max_block_range=1000),
}  # fmt: skip

# Default network for payouts when the agent doesn't pick one: cheapest first.
PAYOUT_PREFERENCE = ["base", "arbitrum", "optimism", "polygon", "ethereum"]


def configured_networks() -> list[Network]:
    """Networks enabled on this clearinghouse (empty when no treasury is set)."""
    if not settings.crypto_treasury_address:
        return []
    spec = (settings.crypto_networks or "").strip()
    if not spec:  # legacy single-network settings
        return [
            Network(
                key=settings.crypto_chain_name.lower().replace(" ", "-"),
                name=settings.crypto_chain_name,
                chain_id=settings.crypto_chain_id,
                rpc_url=settings.crypto_rpc_url,
                token_address=settings.crypto_token_address.lower(),
                token_symbol=settings.crypto_token_symbol,
                decimals=settings.crypto_token_decimals,
                confirmations=settings.crypto_confirmations,
                max_block_range=settings.crypto_max_block_range,
            )
        ]
    if spec.startswith("["):
        networks = []
        for item in json.loads(spec):
            base = PRESETS[item["preset"]] if "preset" in item else None
            fields = {k: v for k, v in item.items() if k != "preset"}
            net = replace(base, **fields) if base else Network(**fields)
            networks.append(replace(net, token_address=net.token_address.lower()))
        return networks
    return [PRESETS[name.strip().lower()] for name in spec.split(",") if name.strip()]


def validate_networks(networks: list[Network]) -> None:
    if not ADDRESS_RE.match(settings.crypto_treasury_address or ""):
        raise RuntimeError("AETHER_CRYPTO_TREASURY_ADDRESS is not a valid 0x address")
    seen = set()
    for net in networks:
        if net.decimals > 9:
            raise RuntimeError(f"{net.name}: tokens with more than 9 decimals are not supported by the nano-USD ledger")
        if net.chain_id in seen:
            raise RuntimeError(f"network chain id {net.chain_id} configured twice")
        seen.add(net.chain_id)


def norm_address(value: str) -> str:
    if not ADDRESS_RE.match(value or ""):
        raise HTTPException(422, detail="not a 0x-prefixed 20-byte address")
    return value.lower()


def topic_for(address: str) -> str:
    return "0x" + "0" * 24 + address[2:].lower()


def address_from_topic(topic: str) -> str:
    return "0x" + topic[-40:].lower()


def chain_of_deposit(deposit_id: str) -> int:
    return int(deposit_id.split(":", 1)[0])


class ChainError(Exception):
    pass


class EvmClient:
    """Minimal JSON-RPC client for the few calls the rails need."""

    def __init__(self, http: httpx.AsyncClient, url: str):
        self.http = http
        self.url = url
        self._id = 0

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

    async def code(self, address: str) -> str:
        return await self.call("eth_getCode", [address, "latest"]) or "0x"

    async def aclose(self) -> None:
        await self.http.aclose()


@dataclass
class Rail:
    """One network's client + deposit watcher."""

    network: Network
    client: EvmClient
    watcher: "DepositWatcher"


class CryptoRails:
    def __init__(self, rails: dict[int, Rail]):
        self.rails = rails

    @classmethod
    def create(cls) -> "CryptoRails | None":
        networks = configured_networks()
        if not networks:
            return None
        validate_networks(networks)
        rails = {}
        for net in networks:
            client = EvmClient(httpx.AsyncClient(timeout=httpx.Timeout(15.0)), net.rpc_url)
            rails[net.chain_id] = Rail(net, client, DepositWatcher(net, client))
        return cls(rails)

    def network(self, chain_id: int) -> Network | None:
        rail = self.rails.get(chain_id)
        return rail.network if rail else None

    def pick(self, choice: str | int | None) -> Rail:
        if choice is None:
            by_key = {r.network.key: r for r in self.rails.values()}
            for key in PAYOUT_PREFERENCE:
                if key in by_key:
                    return by_key[key]
            return next(iter(self.rails.values()))
        for rail in self.rails.values():
            if str(choice).lower() in (rail.network.key, str(rail.network.chain_id), rail.network.name.lower()):
                return rail
        raise HTTPException(
            422, detail=f"unsupported network {choice!r}; use one of {[r.network.key for r in self.rails.values()]}"
        )

    async def aclose(self) -> None:
        for rail in self.rails.values():
            await rail.client.aclose()


def _rails(request: Request) -> CryptoRails:
    rails = getattr(request.app.state, "crypto", None)
    if rails is None:
        raise HTTPException(503, detail="crypto payments are not configured on this clearinghouse")
    return rails


def funding_instructions(rails: "CryptoRails | None" = None) -> dict | None:
    """Machine-readable 'how to pay us', embedded in 402 responses and discovery docs."""
    networks = [r.network for r in rails.rails.values()] if rails else configured_networks()
    if not networks:
        return None
    return {
        "method": "crypto",
        "treasury_address": settings.crypto_treasury_address,
        "networks": [n.public() for n in networks],
        "steps": [
            "GET /v1/billing/crypto/link-challenge?address=<your 0x wallet>",
            "sign the returned message with that wallet (EIP-191 personal_sign)",
            "POST /v1/billing/crypto/wallets {address, nonce, signature}",
            "send native USDC on any listed network from that wallet to treasury_address",
            "balance is credited after the network's confirmations: GET /v1/agents/me",
        ],
    }


def require_admin(x_admin_token: str | None) -> None:
    """Money-moving admin actions always need the admin token, even in sandbox mode."""
    if not settings.admin_token or x_admin_token is None or not hmac.compare_digest(x_admin_token, settings.admin_token):
        raise HTTPException(403, detail="admin token required")


# ------------------------------------------------------------------- deposits


def _deposit_nanos(deposit: CryptoDeposit, network: Network | None) -> int:
    decimals = network.decimals if network else 6
    return deposit.amount_units * (NANOS_PER_USD // 10**decimals)


async def record_deposit(
    network: Network, deposit_id: str, tx_hash: str, from_address: str, units: int, block: int
) -> str | None:
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
                _credit(session, await ledger.lock_agents(session, [wallet.agent_id]), deposit, network)
    except IntegrityError:
        return None  # a concurrent watcher recorded it first
    log.info("DEPOSIT %s %s from %s tx=%s", deposit.status, network.fmt(units), from_address, tx_hash)
    return deposit.status


def _credit(session, agents, deposit: CryptoDeposit, network: Network | None) -> None:
    nanos = _deposit_nanos(deposit, network)
    label = network.fmt(deposit.amount_units) if network else f"{deposit.amount_units} units"
    ledger.post(
        session,
        agents,
        [ledger.house_leg(ledger.HOUSE_CRYPTO_IN, -nanos), ledger.agent_leg(deposit.agent_id, ledger.AVAILABLE, nanos)],
        kind="deposit",
        ref_type="crypto",
        ref_id=deposit.id,
        memo=f"{label} tx {deposit.tx_hash}",
    )


class DepositWatcher:
    def __init__(self, network: Network, client: EvmClient):
        self.network = network
        self.client = client
        self._next_run = 0.0

    async def maybe_poll(self) -> None:
        if time.monotonic() < self._next_run:
            return
        self._next_run = time.monotonic() + settings.crypto_poll_interval_s
        await self.poll()

    async def poll(self, max_batches: int = 20) -> int:
        net = self.network
        treasury = settings.crypto_treasury_address.lower()
        safe = await self.client.block_number() - net.confirmations
        async with SessionLocal() as session, session.begin():
            cursor = await session.get(ChainCursor, net.chain_id)
            if cursor is None:
                start = settings.crypto_start_block if settings.crypto_start_block is not None else safe
                cursor = ChainCursor(chain_id=net.chain_id, last_block=start - 1)
                session.add(cursor)
            last = cursor.last_block
        seen = 0
        for _ in range(max_batches):
            frm = last + 1
            if frm > safe:
                break
            to = min(safe, frm + net.max_block_range - 1)
            for entry in await self.client.transfers_to(net.token_address, treasury, frm, to):
                if entry.get("removed"):
                    continue
                if entry["address"].lower() != net.token_address or entry["topics"][0] != TRANSFER_TOPIC:
                    continue
                deposit_id = f"{net.chain_id}:{entry['transactionHash'].lower()}:{int(entry['logIndex'], 16)}"
                status = await record_deposit(
                    net,
                    deposit_id,
                    entry["transactionHash"].lower(),
                    address_from_topic(entry["topics"][1]),
                    int(entry["data"], 16),
                    int(entry["blockNumber"], 16),
                )
                seen += status is not None
            async with SessionLocal() as session, session.begin():
                cursor = (
                    await session.execute(select(ChainCursor).where(ChainCursor.chain_id == net.chain_id).with_for_update())
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
    network: str | None = Field(None, description="network key or chain id; default: cheapest configured network")


def _link_message(agent_id: str, address: str, nonce: str) -> str:
    return f"Aether wallet link\nagent: {agent_id}\naddress: {address}\nnetworks: all supported EVM networks\nnonce: {nonce}"


def _nonce_key(agent_id: str, nonce: str) -> str:
    return f"{settings.redis_prefix}:cryptolink:{agent_id}:{nonce}"


@router.get("/v1/billing/crypto")
async def deposit_instructions(request: Request, ctx: AuthContext = Depends(authenticate)):
    rails = _rails(request)
    async with SessionLocal() as session:
        wallets = (
            (await session.execute(select(LinkedWallet.address).where(LinkedWallet.agent_id == ctx.agent_id))).scalars().all()
        )
    return funding_instructions(rails) | {
        "linked_wallets": list(wallets),
        "note": "Only native USDC on the listed networks is credited. Other tokens, bridged USDC.e and other networks are not.",
    }


@router.get("/v1/billing/crypto/link-challenge")
async def link_challenge(address: str, request: Request, ctx: AuthContext = Depends(authenticate)):
    _rails(request)
    addr = norm_address(address)
    nonce = secrets.token_hex(16)
    await request.app.state.redis.set(_nonce_key(ctx.agent_id, nonce), addr, ex=600)
    return {"message": _link_message(ctx.agent_id, addr, nonce), "nonce": nonce, "expires_in": 600}


@router.post("/v1/billing/crypto/wallets", status_code=201)
async def link_wallet(body: LinkWalletIn, request: Request, ctx: AuthContext = Depends(authenticate)):
    rails = _rails(request)
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
                    network = rails.network(chain_of_deposit(deposit.id))
                    deposit.agent_id, deposit.status = ctx.agent_id, "credited"
                    _credit(session, agents, deposit, network)
                    credited += _deposit_nanos(deposit, network)
    except IntegrityError:
        raise HTTPException(409, detail="address is linked to another agent") from None
    return {"address": addr, "linked": True, "credited_now_usd": fmt_usd(credited)}


@router.get("/v1/billing/crypto/deposits")
async def my_deposits(request: Request, ctx: AuthContext = Depends(authenticate)):
    rails = getattr(request.app.state, "crypto", None)
    async with SessionLocal() as session:
        rows = (
            (
                await session.execute(
                    select(CryptoDeposit).where(CryptoDeposit.agent_id == ctx.agent_id).order_by(CryptoDeposit.created_at.desc())
                )
            )
            .scalars()
            .all()
        )
    out = []
    for d in rows:
        network = rails.network(chain_of_deposit(d.id)) if rails else None
        out.append(
            {
                "id": d.id,
                "network": network.key if network else chain_of_deposit(d.id),
                "tx_hash": d.tx_hash,
                "amount": network.fmt(d.amount_units) if network else str(d.amount_units),
                "block": d.block_number,
                "status": d.status,
            }
        )
    return out


@router.post("/v1/billing/crypto/withdrawals", status_code=201)
async def request_withdrawal(body: CryptoWithdrawalIn, request: Request, ctx: AuthContext = Depends(authenticate)):
    """Withdraw available balance to one of the agent's own linked wallets."""
    rail = _rails(request).pick(body.network)
    net = rail.network
    if body.amount_usd < settings.crypto_min_withdrawal_usd:
        raise HTTPException(422, detail=f"minimum withdrawal is ${settings.crypto_min_withdrawal_usd}")
    to = norm_address(body.to_address)
    units = int(body.amount_usd * 10**net.decimals)
    nanos = net.to_nanos(units)
    async with SessionLocal() as session, session.begin():
        wallet = await session.get(LinkedWallet, to)
        if wallet is None or wallet.agent_id != ctx.agent_id:
            raise HTTPException(409, detail="payouts go only to a wallet this agent has linked")
        agents = await ledger.lock_agents(session, [ctx.agent_id])
        payout = CryptoPayout(agent_id=ctx.agent_id, to_address=to, amount_units=units, status="pending", chain_id=net.chain_id)
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
    log.info("PAYOUT requested %s agent=%s to=%s id=%s", net.fmt(units), ctx.agent_id, to, payout.id)
    return {"payout_id": payout.id, "status": "pending", "amount": net.fmt(units), "network": net.key, "to_address": to}


@router.get("/v1/billing/crypto/withdrawals")
async def my_withdrawals(request: Request, ctx: AuthContext = Depends(authenticate)):
    rails = getattr(request.app.state, "crypto", None)
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
    return [_payout_out(p, rails) for p in rows]


def _payout_network(p: CryptoPayout, rails: "CryptoRails | None") -> Network | None:
    if rails is None:
        return None
    if p.chain_id is None:  # created before multi-network support: the first configured network
        return next(iter(rails.rails.values())).network
    return rails.network(p.chain_id)


def _payout_out(p: CryptoPayout, rails: "CryptoRails | None") -> dict:
    net = _payout_network(p, rails)
    return {
        "payout_id": p.id,
        "agent_id": p.agent_id,
        "network": net.key if net else p.chain_id,
        "chain_id": net.chain_id if net else p.chain_id,
        "token_contract": net.token_address if net else None,
        "to_address": p.to_address,
        "amount": net.fmt(p.amount_units) if net else str(p.amount_units),
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
async def admin_payouts(request: Request, status: str = "pending", x_admin_token: str | None = Header(None)):
    require_admin(x_admin_token)
    rails = getattr(request.app.state, "crypto", None)
    async with SessionLocal() as session:
        rows = (
            (await session.execute(select(CryptoPayout).where(CryptoPayout.status == status).order_by(CryptoPayout.created_at)))
            .scalars()
            .all()
        )
    return {"payouts": [_payout_out(p, rails) for p in rows]}


async def _verify_transfer(rail: Rail, tx_hash: str, to: str, units: int) -> None:
    net, client = rail.network, rail.client
    receipt = await client.receipt(tx_hash)
    if not receipt:
        raise HTTPException(409, detail=f"transaction not found on {net.name} (not mined yet, or wrong network?)")
    if receipt.get("status") != "0x1":
        raise HTTPException(409, detail="transaction failed on-chain")
    confirmations = await client.block_number() - int(receipt["blockNumber"], 16) + 1
    if confirmations < net.confirmations:
        raise HTTPException(409, detail=f"only {confirmations}/{net.confirmations} confirmations; retry shortly")
    treasury = settings.crypto_treasury_address.lower()
    for entry in receipt.get("logs", []):
        topics = entry.get("topics", [])
        if (
            entry.get("address", "").lower() == net.token_address
            and len(topics) == 3
            and topics[0] == TRANSFER_TOPIC
            and address_from_topic(topics[1]) == treasury
            and address_from_topic(topics[2]) == to
            and int(entry["data"], 16) == units
        ):
            return
    raise HTTPException(409, detail=f"no transfer of {net.fmt(units)} from the treasury to {to} in that transaction")


@router.post("/v1/admin/crypto/payouts/{payout_id}/paid")
async def admin_mark_paid(payout_id: str, body: MarkPaidIn, request: Request, x_admin_token: str | None = Header(None)):
    require_admin(x_admin_token)
    rails = _rails(request)
    async with SessionLocal() as session:
        payout = await session.get(CryptoPayout, payout_id)
    if payout is None:
        raise HTTPException(404, detail="payout not found")
    if payout.status != "pending":
        raise HTTPException(409, detail=f"payout is already {payout.status}")
    net = _payout_network(payout, rails)
    if net is None:
        raise HTTPException(409, detail=f"network {payout.chain_id} is not configured on this clearinghouse")
    tx_hash = body.tx_hash.lower()
    await _verify_transfer(rails.rails[net.chain_id], tx_hash, payout.to_address, payout.amount_units)
    nanos = net.to_nanos(payout.amount_units)
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
                memo=f"{net.key} tx {tx_hash}",
            )
    except IntegrityError:
        raise HTTPException(409, detail="that transaction is already recorded for another payout") from None
    log.info("PAYOUT paid %s id=%s tx=%s", net.fmt(payout.amount_units), payout_id, tx_hash)
    return _payout_out(payout, rails)


@router.post("/v1/admin/crypto/payouts/{payout_id}/cancel")
async def admin_cancel(payout_id: str, request: Request, x_admin_token: str | None = Header(None)):
    require_admin(x_admin_token)
    rails = getattr(request.app.state, "crypto", None)
    async with SessionLocal() as session, session.begin():
        payout = (
            await session.execute(select(CryptoPayout).where(CryptoPayout.id == payout_id).with_for_update())
        ).scalar_one_or_none()
        if payout is None:
            raise HTTPException(404, detail="payout not found")
        if payout.status != "pending":
            raise HTTPException(409, detail=f"payout is already {payout.status}")
        net = _payout_network(payout, rails)
        decimals = net.decimals if net else 6
        nanos = payout.amount_units * (NANOS_PER_USD // 10**decimals)
        agents = await ledger.lock_agents(session, [payout.agent_id])
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
    return _payout_out(payout, rails)


@router.get("/v1/admin/crypto/solvency")
async def admin_solvency(request: Request, x_admin_token: str | None = Header(None)):
    """Is the treasury's on-chain balance (all networks) enough to cover every obligation?"""
    require_admin(x_admin_token)
    rails = _rails(request)
    treasury = settings.crypto_treasury_address
    per_network, onchain, errors = [], 0, []
    for rail in rails.rails.values():
        net = rail.network
        try:
            units = await rail.client.token_balance(net.token_address, treasury)
            is_contract = (await rail.client.code(treasury)) not in ("0x", "0x0")
            per_network.append({"network": net.key, "balance": net.fmt(units), "treasury_is_contract": is_contract})
            onchain += net.to_nanos(units)
        except ChainError as exc:
            errors.append(f"{net.key}: {exc}")
            per_network.append({"network": net.key, "balance": None, "error": str(exc)})
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
            await session.execute(select(func.count()).select_from(CryptoDeposit).where(CryptoDeposit.status == "unattributed"))
        ).scalar_one()
    owed_agents = int(balances)
    payable = int(by_account.get(ledger.HOUSE_CRYPTO_PAYABLE, 0))
    obligations = owed_agents + payable
    return {
        "treasury_address": treasury,
        "networks": per_network,
        "onchain_total_usd": fmt_usd(onchain),
        "owed_to_agents_usd": fmt_usd(owed_agents),
        "pending_payouts_usd": fmt_usd(payable),
        "fees_earned_usd": fmt_usd(int(by_account.get(ledger.HOUSE_FEES, 0))),
        "unattributed_deposits": int(unattributed),
        "deposited_total_usd": fmt_usd(-int(by_account.get(ledger.HOUSE_CRYPTO_IN, 0))),
        "paid_out_total_usd": fmt_usd(int(by_account.get(ledger.HOUSE_CRYPTO_OUT, 0))),
        "surplus_usd": fmt_usd(onchain - obligations),
        "solvent": not errors and onchain >= obligations,
        "errors": errors,
    }


async def run_watcher_pass(app) -> None:
    rails: CryptoRails | None = getattr(app.state, "crypto", None)
    if rails is None:
        return
    for rail in rails.rails.values():
        try:
            await rail.watcher.maybe_poll()
        except ChainError as exc:
            log.warning("deposit watcher %s: %s", rail.network.key, exc)


__all__ = [
    "CryptoRails",
    "DepositWatcher",
    "EvmClient",
    "Network",
    "PRESETS",
    "configured_networks",
    "funding_instructions",
    "record_deposit",
    "require_admin",
    "router",
    "run_watcher_pass",
]
