"""Real-money rails on Stripe.

  Deposits   Stripe Checkout -> signed webhook `checkout.session.completed`
             -> ledger: house:stripe_deposits -> agent available (idempotent
             on the Checkout Session id).
  Payouts    Stripe Connect Express. A seller onboards once (Stripe runs the
             KYC), then withdraws: ledger debits available first, then a
             Stripe Transfer moves the money to the connected account. A
             failed transfer is reversed in the ledger.

The platform's revenue is the `house:fees` account (clearing fee on every
settlement), which stays in the platform's Stripe balance.

Stripe is called over its REST API with httpx (no SDK dependency); the
client is on app.state.stripe so tests can swap in a mock transport.
"""

import hashlib
import hmac
import json
import logging
import time
from decimal import Decimal
from typing import Any
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import select

from . import ledger
from .auth import SCOPE_SELL, AuthContext, authenticate, require_scopes
from .config import settings
from .db import SessionLocal
from .models import ExternalPayment, PayoutAccount, Withdrawal, new_id
from .units import NANOS_PER_USD, fmt_usd

log = logging.getLogger("aether.payments")

router = APIRouter(prefix="/v1/billing", tags=["billing"])

NANOS_PER_CENT = NANOS_PER_USD // 100


class StripeError(Exception):
    pass


def _flatten(data: dict[str, Any], prefix: str = "") -> list[tuple[str, str]]:
    """Stripe's form encoding: {"a": {"b": [1]}} -> a[b][0]=1."""
    out: list[tuple[str, str]] = []
    for key, value in data.items():
        name = f"{prefix}[{key}]" if prefix else str(key)
        if isinstance(value, dict):
            out.extend(_flatten(value, name))
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    out.extend(_flatten(item, f"{name}[{i}]"))
                else:
                    out.append((f"{name}[{i}]", str(item)))
        elif isinstance(value, bool):
            out.append((name, "true" if value else "false"))
        elif value is not None:
            out.append((name, str(value)))
    return out


class StripeClient:
    def __init__(self, http: httpx.AsyncClient):
        self.http = http

    @classmethod
    def create(cls) -> "StripeClient | None":
        if not settings.stripe_secret_key:
            return None
        return cls(
            httpx.AsyncClient(
                base_url=settings.stripe_api_base,
                auth=(settings.stripe_secret_key, ""),
                timeout=httpx.Timeout(20.0),
                headers={"Stripe-Version": "2024-06-20"},
            )
        )

    async def post(self, path: str, data: dict[str, Any], idempotency_key: str | None = None) -> dict:
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        resp = await self.http.post(path, content=urlencode(_flatten(data)), headers=headers)
        body = resp.json()
        if resp.status_code >= 400:
            raise StripeError(body.get("error", {}).get("message", f"HTTP {resp.status_code}"))
        return body

    async def aclose(self) -> None:
        await self.http.aclose()


def _stripe(request: Request) -> StripeClient:
    client = getattr(request.app.state, "stripe", None)
    if client is None:
        raise HTTPException(503, detail="payments are not configured on this clearinghouse")
    return client


def verify_stripe_signature(payload: bytes, header: str, secret: str, tolerance_s: int = 300) -> None:
    """Stripe webhook signature scheme v1: HMAC-SHA256 over '<t>.<payload>'."""
    parts = dict(item.split("=", 1) for item in header.split(",") if "=" in item)
    signatures = [v for k, v in (item.split("=", 1) for item in header.split(",") if "=" in item) if k == "v1"]
    try:
        timestamp = int(parts["t"])
    except (KeyError, ValueError):
        raise HTTPException(400, detail="malformed Stripe-Signature") from None
    if abs(time.time() - timestamp) > tolerance_s:
        raise HTTPException(400, detail="stale Stripe webhook")
    expected = hmac.new(secret.encode(), f"{timestamp}.".encode() + payload, hashlib.sha256).hexdigest()
    if not any(hmac.compare_digest(expected, sig) for sig in signatures):
        raise HTTPException(400, detail="invalid Stripe signature")


# ---------------------------------------------------------------------- deposits


class DepositIn(BaseModel):
    amount_usd: Decimal = Field(gt=0, decimal_places=2)


@router.post("/deposits")
async def create_deposit(body: DepositIn, request: Request, ctx: AuthContext = Depends(authenticate)):
    """Returns a Stripe Checkout URL. The agent hands it to its operator (or a
    payment-capable agent completes it); the balance is credited by webhook."""
    stripe = _stripe(request)
    if not settings.min_deposit_usd <= body.amount_usd <= settings.max_deposit_usd:
        raise HTTPException(422, detail=f"deposit must be between ${settings.min_deposit_usd} and ${settings.max_deposit_usd}")
    cents = int(body.amount_usd * 100)
    try:
        session = await stripe.post(
            "/v1/checkout/sessions",
            {
                "mode": "payment",
                "client_reference_id": ctx.agent_id,
                "metadata": {"agent_id": ctx.agent_id},
                "line_items": [
                    {
                        "quantity": 1,
                        "price_data": {
                            "currency": "usd",
                            "unit_amount": cents,
                            "product_data": {"name": "Aether inference credit"},
                        },
                    }
                ],
                "success_url": f"{settings.public_base_url}/v1/billing/return?status=success",
                "cancel_url": f"{settings.public_base_url}/v1/billing/return?status=cancelled",
            },
            idempotency_key=new_id("dep"),
        )
    except StripeError as exc:
        raise HTTPException(502, detail=f"Stripe: {exc}") from None
    return {"checkout_url": session["url"], "session_id": session["id"], "amount_usd": str(body.amount_usd)}


@router.get("/return")
async def checkout_return(status: str = "success"):
    return {"status": status, "detail": "balance is credited when Stripe confirms payment"}


async def credit_deposit(session_id: str, agent_id: str, amount_nanos: int) -> bool:
    """Idempotent: a replayed webhook for the same session credits nothing."""
    async with SessionLocal() as session, session.begin():
        if await session.get(ExternalPayment, session_id) is not None:
            return False
        session.add(ExternalPayment(id=session_id, agent_id=agent_id, provider="stripe", amount_nanos=amount_nanos))
        await session.flush()
        agents = await ledger.lock_agents(session, [agent_id])
        ledger.post(
            session,
            agents,
            [ledger.house_leg(ledger.HOUSE_STRIPE_IN, -amount_nanos), ledger.agent_leg(agent_id, ledger.AVAILABLE, amount_nanos)],
            kind="deposit",
            ref_type="stripe",
            ref_id=session_id,
            memo="Stripe Checkout deposit",
        )
    log.info("DEPOSIT agent=%s amount=%s session=%s", agent_id, fmt_usd(amount_nanos), session_id)
    return True


@router.post("/stripe/webhook", include_in_schema=False)
async def stripe_webhook(request: Request, stripe_signature: str = Header(...)):
    if not settings.stripe_webhook_secret:
        raise HTTPException(503, detail="webhook secret not configured")
    payload = await request.body()
    verify_stripe_signature(payload, stripe_signature, settings.stripe_webhook_secret)
    event = json.loads(payload)
    obj = event.get("data", {}).get("object", {})
    if event.get("type") in ("checkout.session.completed", "checkout.session.async_payment_succeeded"):
        if obj.get("payment_status") == "paid" and obj.get("currency", "usd") == "usd":
            agent_id = (obj.get("metadata") or {}).get("agent_id") or obj.get("client_reference_id")
            try:
                await credit_deposit(obj["id"], agent_id, int(obj["amount_total"]) * NANOS_PER_CENT)
            except ledger.LedgerError:
                log.exception("could not credit Stripe session %s", obj.get("id"))
                raise HTTPException(500, detail="credit failed; Stripe will retry") from None
    return {"received": True}


# ----------------------------------------------------------------------- payouts


@router.post("/connect/onboard")
async def connect_onboard(request: Request, ctx: AuthContext = Depends(require_scopes(SCOPE_SELL))):
    """Create (once) the seller's Stripe Express account and return an
    onboarding link for its operator. Stripe performs identity verification."""
    stripe = _stripe(request)
    async with SessionLocal() as session:
        account = await session.get(PayoutAccount, ctx.agent_id)
    try:
        if account is None:
            created = await stripe.post(
                "/v1/accounts",
                {
                    "type": "express",
                    "capabilities": {"transfers": {"requested": True}},
                    "metadata": {"agent_id": ctx.agent_id},
                },
                idempotency_key=f"acct-{ctx.agent_id}",
            )
            async with SessionLocal() as session, session.begin():
                account = PayoutAccount(agent_id=ctx.agent_id, stripe_account_id=created["id"])
                session.add(account)
        link = await stripe.post(
            "/v1/account_links",
            {
                "account": account.stripe_account_id,
                "type": "account_onboarding",
                "refresh_url": f"{settings.public_base_url}/v1/billing/return?status=refresh",
                "return_url": f"{settings.public_base_url}/v1/billing/return?status=onboarded",
            },
        )
    except StripeError as exc:
        raise HTTPException(502, detail=f"Stripe: {exc}") from None
    return {"onboarding_url": link["url"], "stripe_account_id": account.stripe_account_id}


class WithdrawalIn(BaseModel):
    amount_usd: Decimal = Field(gt=0, decimal_places=2)


@router.post("/withdrawals")
async def create_withdrawal(body: WithdrawalIn, request: Request, ctx: AuthContext = Depends(require_scopes(SCOPE_SELL))):
    stripe = _stripe(request)
    if body.amount_usd < settings.min_withdrawal_usd:
        raise HTTPException(422, detail=f"minimum withdrawal is ${settings.min_withdrawal_usd}")
    amount = int(body.amount_usd * 100) * NANOS_PER_CENT
    withdrawal_id = new_id("wd")
    async with SessionLocal() as session, session.begin():
        account = await session.get(PayoutAccount, ctx.agent_id)
        if account is None:
            raise HTTPException(409, detail="complete payout onboarding first: POST /v1/billing/connect/onboard")
        agents = await ledger.lock_agents(session, [ctx.agent_id])
        try:
            ledger.post(
                session,
                agents,
                [ledger.agent_leg(ctx.agent_id, ledger.AVAILABLE, -amount), ledger.house_leg(ledger.HOUSE_STRIPE_OUT, amount)],
                kind="withdrawal",
                ref_type="withdrawal",
                ref_id=withdrawal_id,
            )
        except ledger.InsufficientFunds:
            raise HTTPException(402, detail="insufficient available balance") from None
        session.add(Withdrawal(id=withdrawal_id, agent_id=ctx.agent_id, amount_nanos=amount, status="pending"))
        destination = account.stripe_account_id

    try:
        transfer = await stripe.post(
            "/v1/transfers",
            {
                "amount": amount // NANOS_PER_CENT,
                "currency": "usd",
                "destination": destination,
                "metadata": {"agent_id": ctx.agent_id, "withdrawal_id": withdrawal_id},
            },
            idempotency_key=withdrawal_id,
        )
    except (StripeError, httpx.HTTPError) as exc:
        async with SessionLocal() as session, session.begin():
            agents = await ledger.lock_agents(session, [ctx.agent_id])
            ledger.post(
                session,
                agents,
                [ledger.house_leg(ledger.HOUSE_STRIPE_OUT, -amount), ledger.agent_leg(ctx.agent_id, ledger.AVAILABLE, amount)],
                kind="withdrawal_reversal",
                ref_type="withdrawal",
                ref_id=withdrawal_id,
                memo=str(exc)[:250],
            )
            row = await session.get(Withdrawal, withdrawal_id)
            row.status, row.error = "failed", str(exc)
        raise HTTPException(502, detail=f"payout failed and was reversed: {exc}") from None

    async with SessionLocal() as session, session.begin():
        row = await session.get(Withdrawal, withdrawal_id)
        row.status, row.stripe_transfer_id = "paid", transfer["id"]
    log.info("PAYOUT agent=%s amount=%s transfer=%s", ctx.agent_id, fmt_usd(amount), transfer["id"])
    return {"withdrawal_id": withdrawal_id, "status": "paid", "amount_usd": str(body.amount_usd), "transfer_id": transfer["id"]}


@router.get("/withdrawals")
async def list_withdrawals(ctx: AuthContext = Depends(authenticate)):
    async with SessionLocal() as session:
        rows = (
            (
                await session.execute(
                    select(Withdrawal).where(Withdrawal.agent_id == ctx.agent_id).order_by(Withdrawal.created_at.desc())
                )
            )
            .scalars()
            .all()
        )
    return [
        {"withdrawal_id": w.id, "amount_usd": fmt_usd(w.amount_nanos), "status": w.status, "transfer_id": w.stripe_transfer_id}
        for w in rows
    ]
