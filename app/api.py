"""Agent registry, order entry, market data, allocations and ledger endpoints."""

import asyncio
import hmac
import time
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Literal
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from . import house, ledger, netguard
from .auth import (
    ALL_SCOPES,
    SCOPE_BUY,
    SCOPE_SELL,
    AuthContext,
    authenticate,
    check_scopes,
    generate_client_credentials,
    hash_secret,
    make_api_key,
    require_scopes,
)
from .config import settings
from .crypto_payments import funding_instructions
from .db import SessionLocal, get_session, utcnow
from .exchange import ExchangeError, cancel_order, place_order, release_trade
from .models import (
    ASK,
    BID,
    GTC,
    IOC,
    LIMIT,
    MARKET,
    ORDER_OPEN,
    TRADE_ACTIVE,
    Agent,
    Feedback,
    InferenceJob,
    Order,
    Service,
    ServiceCall,
    Trade,
    TradeLedger,
)
from .units import fmt_usd, npt_to_usd_per_mtok, usd_per_mtok_to_npt, usd_to_nanos

router = APIRouter(prefix="/v1")

INSTRUMENT_PATTERN = r"^[a-z0-9][a-z0-9._:-]{1,63}$"


def _raise(exc: ExchangeError):
    detail = {"error": "rejected", "error_description": exc.message}
    if exc.status_code == 402 and (how := funding_instructions()):
        detail["how_to_fund"] = how  # machine-actionable: exactly how to pay
    raise HTTPException(exc.status_code, detail=detail)


# ------------------------------------------------------------------- schemas


class RegisterAgentIn(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    scopes: list[str] = Field(default_factory=lambda: [SCOPE_BUY])

    @field_validator("scopes")
    @classmethod
    def _known_scopes(cls, v: list[str]) -> list[str]:
        unknown = set(v) - ALL_SCOPES
        if unknown or not v:
            raise ValueError(f"scopes must be a non-empty subset of {sorted(ALL_SCOPES)}")
        return sorted(set(v))


class RegisterAgentOut(BaseModel):
    agent_id: str
    client_id: str
    client_secret: str = Field(description="shown once; only a scrypt hash is stored")
    api_key: str = Field(description="'<client_id>.<client_secret>' for OpenAI-style clients; shown once")
    scopes: list[str]


class AgentOut(BaseModel):
    agent_id: str
    name: str
    client_id: str
    scopes: list[str]
    endpoint_url: str | None
    available_usd: str
    escrow_usd: str
    available_nanos: int
    escrow_nanos: int


class EndpointIn(BaseModel):
    endpoint_url: str = Field(max_length=512)


class FaucetIn(BaseModel):
    amount_usd: Decimal = Field(gt=0)


class OrderIn(BaseModel):
    instrument: str = Field(pattern=INSTRUMENT_PATTERN, description="model/product being traded")
    side: Literal["bid", "ask"]
    order_type: Literal["limit", "market"] = LIMIT
    time_in_force: Literal["gtc", "ioc"] = GTC
    price_usd_per_mtok: Decimal = Field(
        gt=0, description="limit price; for market orders the protection cap (bid) or floor (ask)"
    )
    quantity_tokens: int = Field(gt=0, le=settings.max_order_tokens)
    ttl_seconds: int | None = Field(None, gt=0, le=30 * 86400)


class OrderOut(BaseModel):
    order_id: str
    instrument: str
    side: str
    order_type: str
    time_in_force: str
    price_usd_per_mtok: str
    quantity_tokens: int
    filled_tokens: int
    cancelled_tokens: int
    open_tokens: int
    escrow_usd: str
    status: str
    expires_at: datetime | None
    created_at: datetime


class TradeOut(BaseModel):
    trade_id: str
    instrument: str
    role: str
    counterparty_id: str
    price_usd_per_mtok: str
    tokens_total: int
    tokens_used: int
    tokens_reserved: int
    tokens_available: int
    escrow_usd: str
    settled_usd: str
    fee_usd: str
    status: str
    expires_at: datetime


class PlaceOrderOut(BaseModel):
    order: OrderOut
    fills: list[TradeOut]


def order_out(o: Order) -> OrderOut:
    return OrderOut(
        order_id=o.id,
        instrument=o.instrument,
        side=o.side,
        order_type=o.order_type,
        time_in_force=o.time_in_force,
        price_usd_per_mtok=npt_to_usd_per_mtok(o.price_npt),
        quantity_tokens=o.quantity_tokens,
        filled_tokens=o.filled_tokens,
        cancelled_tokens=o.cancelled_tokens,
        open_tokens=o.open_tokens,
        escrow_usd=fmt_usd(o.escrow_nanos),
        status=o.status,
        expires_at=o.expires_at,
        created_at=o.created_at,
    )


def trade_out(t: Trade, viewer_id: str) -> TradeOut:
    buyer = t.buyer_id == viewer_id
    return TradeOut(
        trade_id=t.id,
        instrument=t.instrument,
        role="buyer" if buyer else "seller",
        counterparty_id=t.seller_id if buyer else t.buyer_id,
        price_usd_per_mtok=npt_to_usd_per_mtok(t.price_npt),
        tokens_total=t.tokens_total,
        tokens_used=t.tokens_used,
        tokens_reserved=t.tokens_reserved,
        tokens_available=t.tokens_available,
        escrow_usd=fmt_usd(t.escrow_nanos),
        settled_usd=fmt_usd(t.gross_settled_nanos),
        fee_usd=fmt_usd(t.fee_nanos),
        status=t.status,
        expires_at=t.expires_at,
    )


def agent_out(a: Agent) -> AgentOut:
    return AgentOut(
        agent_id=a.id,
        name=a.name,
        client_id=a.client_id,
        scopes=sorted(a.scopes),
        endpoint_url=a.endpoint_url,
        available_usd=fmt_usd(a.balance_available_nanos),
        escrow_usd=fmt_usd(a.balance_escrow_nanos),
        available_nanos=a.balance_available_nanos,
        escrow_nanos=a.balance_escrow_nanos,
    )


# -------------------------------------------------------------------- agents


@router.post("/agents", response_model=RegisterAgentOut, status_code=201, tags=["agents"])
async def register_agent(
    body: RegisterAgentIn,
    request: Request,
    session: AsyncSession = Depends(get_session),
    x_admin_token: str | None = Header(None),
):
    """Self-serve signup for agents. New agents start with a zero balance; buyers
    fund via Stripe deposits, sellers are paid out only after Stripe KYC."""
    is_admin = (
        bool(settings.admin_token) and x_admin_token is not None and hmac.compare_digest(x_admin_token, settings.admin_token)
    )
    if not (settings.sandbox_mode or settings.open_registration or is_admin):
        raise HTTPException(403, detail="registration is invite-only on this clearinghouse (X-Admin-Token)")
    if not is_admin and settings.registrations_per_ip_per_hour > 0:
        ip = request.client.host if request.client else "unknown"
        key = f"{settings.redis_prefix}:rl:register:{ip}:{int(time.time() // 3600)}"
        pipe = request.app.state.redis.pipeline(transaction=True)
        pipe.incr(key)
        pipe.expire(key, 3700)
        count, _ = await pipe.execute()
        if count > settings.registrations_per_ip_per_hour:
            raise HTTPException(429, detail="too many registrations from this address; try again later")
    client_id, client_secret = generate_client_credentials()
    secret_hash = await asyncio.to_thread(hash_secret, client_secret)
    agent = Agent(
        name=body.name,
        client_id=client_id,
        client_secret_hash=secret_hash,
        allowed_scopes=" ".join(body.scopes),
        balance_available_nanos=0,
        balance_escrow_nanos=0,
        token_version=0,
        is_active=True,
    )
    session.add(agent)
    await session.commit()
    return RegisterAgentOut(
        agent_id=agent.id,
        client_id=client_id,
        client_secret=client_secret,
        api_key=make_api_key(client_id, client_secret),
        scopes=body.scopes,
    )


@router.post("/agents/me/rotate-secret", response_model=RegisterAgentOut, tags=["agents"])
async def rotate_secret(ctx: AuthContext = Depends(authenticate)):
    """Issue a new client secret. The old secret, API key and every JWT issued
    so far stop working immediately."""
    _, client_secret = generate_client_credentials()
    secret_hash = await asyncio.to_thread(hash_secret, client_secret)
    async with SessionLocal() as session, session.begin():
        agent = (await session.execute(select(Agent).where(Agent.id == ctx.agent_id).with_for_update())).scalar_one()
        agent.client_secret_hash = secret_hash
        agent.token_version += 1
    return RegisterAgentOut(
        agent_id=agent.id,
        client_id=agent.client_id,
        client_secret=client_secret,
        api_key=make_api_key(agent.client_id, client_secret),
        scopes=sorted(agent.scopes),
    )


@router.get("/agents/me", response_model=AgentOut, tags=["agents"])
async def me(ctx: AuthContext = Depends(authenticate), session: AsyncSession = Depends(get_session)):
    return agent_out(await session.get(Agent, ctx.agent_id))


def _is_own_house_endpoint(url: str) -> bool:
    """One exemption from the address rules: the platform's own house services
    (app/house.py) are listed like any other seller, but their endpoint is this
    process' public URL — which the clearinghouse's own resolver answers with a
    private or CGNAT address. The origin comes from operator config, never from
    an agent, and only the house paths qualify."""
    own, given = urlparse(settings.public_base_url), urlparse(url)
    return (own.scheme, own.netloc) == (given.scheme, given.netloc) and given.path.startswith(house.HOUSE_PREFIX + "/")


async def _validate_endpoint(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise HTTPException(422, detail="endpoint_url must be an absolute http(s) URL")
    if settings.allow_private_seller_urls or _is_own_house_endpoint(url):
        return
    try:
        await netguard.check_public_url(url)
    except netguard.UnsafeURL as exc:
        raise HTTPException(422, detail=f"endpoint_url {exc}") from None


@router.put("/agents/me/endpoint", response_model=AgentOut, tags=["agents"])
async def set_endpoint(
    body: EndpointIn,
    ctx: AuthContext = Depends(require_scopes(SCOPE_SELL)),
    session: AsyncSession = Depends(get_session),
):
    """Register the seller's stateless inference webhook. The clearinghouse is
    the only caller; each call carries a single-use delivery JWT."""
    await _validate_endpoint(body.endpoint_url)
    agent = await session.get(Agent, ctx.agent_id)
    agent.endpoint_url = body.endpoint_url
    await session.commit()
    return agent_out(agent)


@router.post("/sandbox/faucet", response_model=AgentOut, tags=["sandbox"])
async def faucet(body: FaucetIn, ctx: AuthContext = Depends(authenticate)):
    if not settings.sandbox_mode:
        raise HTTPException(404, detail="not found")
    if body.amount_usd > Decimal(str(settings.max_faucet_usd)):
        raise HTTPException(422, detail=f"max faucet amount is ${settings.max_faucet_usd}")
    try:
        amount = usd_to_nanos(body.amount_usd)
    except ValueError as exc:
        raise HTTPException(422, detail=str(exc)) from None
    async with SessionLocal() as session, session.begin():
        agents = await ledger.lock_agents(session, [ctx.agent_id])
        ledger.mint(session, agents, ctx.agent_id, amount, memo="sandbox faucet")
    return agent_out(agents[ctx.agent_id])


# -------------------------------------------------------------------- orders


@router.post("/orders", response_model=PlaceOrderOut, status_code=201, tags=["market"])
async def create_order(body: OrderIn, request: Request, ctx: AuthContext = Depends(authenticate)):
    check_scopes(ctx, SCOPE_BUY if body.side == BID else SCOPE_SELL)
    try:
        price_npt = usd_per_mtok_to_npt(body.price_usd_per_mtok)
    except ValueError as exc:
        raise HTTPException(422, detail=str(exc)) from None
    try:
        order, trades = await place_order(
            request.app.state.engine,
            request.app.state.redis,
            agent_id=ctx.agent_id,
            instrument=body.instrument,
            side=body.side,
            order_type=body.order_type,
            time_in_force=IOC if body.order_type == MARKET else body.time_in_force,
            price_npt=price_npt,
            quantity=body.quantity_tokens,
            ttl_s=body.ttl_seconds,
        )
    except ExchangeError as exc:
        _raise(exc)
    return PlaceOrderOut(order=order_out(order), fills=[trade_out(t, ctx.agent_id) for t in trades])


@router.get("/orders", response_model=list[OrderOut], tags=["market"])
async def list_orders(
    status: str | None = Query(None),
    limit: int = Query(50, le=500),
    ctx: AuthContext = Depends(authenticate),
    session: AsyncSession = Depends(get_session),
):
    q = select(Order).where(Order.agent_id == ctx.agent_id)
    if status:
        q = q.where(Order.status == status)
    rows = await session.execute(q.order_by(Order.created_at.desc()).limit(limit))
    return [order_out(o) for o in rows.scalars()]


@router.get("/orders/{order_id}", response_model=OrderOut, tags=["market"])
async def get_order(order_id: str, ctx: AuthContext = Depends(authenticate), session: AsyncSession = Depends(get_session)):
    order = await session.get(Order, order_id)
    if order is None or order.agent_id != ctx.agent_id:
        raise HTTPException(404, detail="order not found")
    return order_out(order)


@router.delete("/orders/{order_id}", response_model=OrderOut, tags=["market"])
async def delete_order(order_id: str, request: Request, ctx: AuthContext = Depends(authenticate)):
    try:
        return order_out(await cancel_order(request.app.state.engine, agent_id=ctx.agent_id, order_id=order_id))
    except ExchangeError as exc:
        _raise(exc)


@router.get("/book/{instrument}", tags=["market"])
async def order_book(instrument: str, request: Request, levels: int = Query(10, ge=1, le=100)):
    depth = await request.app.state.engine.depth(instrument, levels)

    def fmt(levels_):
        return [{"price_usd_per_mtok": npt_to_usd_per_mtok(p), "tokens": q, "orders": n} for p, q, n in levels_]

    return {"instrument": instrument, "bids": fmt(depth["bids"]), "asks": fmt(depth["asks"])}


@router.get("/market/{instrument}/stream", tags=["market"])
async def market_stream(instrument: str, request: Request):
    """Server-sent trade prints (Redis Pub/Sub fan-out)."""
    pubsub = request.app.state.redis.pubsub()
    await pubsub.subscribe(f"{settings.redis_prefix}:md:{instrument}")

    async def events():
        try:
            while not await request.is_disconnected():
                msg = await pubsub.get_message(ignore_subscribe_messages=True, timeout=15.0)
                yield f"data: {msg['data']}\n\n" if msg else ": keep-alive\n\n"
        finally:
            await pubsub.unsubscribe()
            await pubsub.aclose()

    return StreamingResponse(events(), media_type="text/event-stream")


# ------------------------------------------------------ allocations & ledger


@router.get("/trades", response_model=list[TradeOut], tags=["allocations"])
async def list_trades(
    active_only: bool = False,
    limit: int = Query(100, le=500),
    ctx: AuthContext = Depends(authenticate),
    session: AsyncSession = Depends(get_session),
):
    q = select(Trade).where(or_(Trade.buyer_id == ctx.agent_id, Trade.seller_id == ctx.agent_id))
    if active_only:
        q = q.where(Trade.status == TRADE_ACTIVE)
    rows = await session.execute(q.order_by(Trade.created_at.desc()).limit(limit))
    return [trade_out(t, ctx.agent_id) for t in rows.scalars()]


@router.post("/trades/{trade_id}/release", response_model=TradeOut, tags=["allocations"])
async def release_allocation(trade_id: str, ctx: AuthContext = Depends(require_scopes(SCOPE_BUY))):
    try:
        return trade_out(await release_trade(agent_id=ctx.agent_id, trade_id=trade_id), ctx.agent_id)
    except ExchangeError as exc:
        _raise(exc)


@router.get("/ledger", tags=["ledger"])
async def my_ledger(
    limit: int = Query(100, le=1000),
    ctx: AuthContext = Depends(authenticate),
    session: AsyncSession = Depends(get_session),
):
    rows = await session.execute(
        select(TradeLedger).where(TradeLedger.agent_id == ctx.agent_id).order_by(TradeLedger.id.desc()).limit(limit)
    )
    return [
        {
            "id": e.id,
            "tx_id": e.tx_id,
            "account": e.account,
            "amount_nanos": e.amount_nanos,
            "amount_usd": fmt_usd(e.amount_nanos),
            "kind": e.kind,
            "ref": f"{e.ref_type}:{e.ref_id}",
            "memo": e.memo,
            "created_at": e.created_at,
        }
        for e in rows.scalars()
    ]


def _admin_or_sandbox(x_admin_token: str | None) -> None:
    is_admin = (
        bool(settings.admin_token) and x_admin_token is not None and hmac.compare_digest(x_admin_token, settings.admin_token)
    )
    if not (settings.sandbox_mode or is_admin):
        raise HTTPException(404, detail="not found")


@router.get("/admin/stats", tags=["admin"])
async def admin_stats(session: AsyncSession = Depends(get_session), x_admin_token: str | None = Header(None)):
    """Operator dashboard numbers: revenue, volume, money in/out, activity."""
    _admin_or_sandbox(x_admin_token)
    by_account = dict(
        (
            await session.execute(select(TradeLedger.account, func.sum(TradeLedger.amount_nanos)).group_by(TradeLedger.account))
        ).all()
    )
    since = utcnow() - timedelta(hours=24)
    jobs = dict(
        (
            await session.execute(
                select(InferenceJob.status, func.count()).where(InferenceJob.created_at >= since).group_by(InferenceJob.status)
            )
        ).all()
    )
    trades = (
        await session.execute(
            select(
                func.count(), func.coalesce(func.sum(Trade.gross_settled_nanos), 0), func.coalesce(func.sum(Trade.tokens_used), 0)
            )
        )
    ).one()
    agents = (await session.execute(select(func.count()).select_from(Agent))).scalar_one()
    sellers = (await session.execute(select(func.count()).select_from(Agent).where(Agent.endpoint_url.is_not(None)))).scalar_one()
    return {
        "revenue_fees_usd": fmt_usd(int(by_account.get(ledger.HOUSE_FEES, 0))),
        "settled_volume_usd": fmt_usd(int(trades[1])),
        "tokens_delivered": int(trades[2]),
        "trades": int(trades[0]),
        "deposits_usd": fmt_usd(-int(by_account.get(ledger.HOUSE_STRIPE_IN, 0))),
        "payouts_usd": fmt_usd(int(by_account.get(ledger.HOUSE_STRIPE_OUT, 0))),
        "sandbox_minted_usd": fmt_usd(-int(by_account.get(ledger.HOUSE_MINT, 0))),
        "agents": agents,
        "sellers_with_endpoint": sellers,
        "jobs_24h": jobs,
        "services_active": (
            await session.execute(select(func.count()).select_from(Service).where(Service.status == "active"))
        ).scalar_one(),
        "service_calls": dict(
            (await session.execute(select(ServiceCall.status, func.count()).group_by(ServiceCall.status))).all()
        ),
        "feedback_24h": (
            await session.execute(select(func.count()).select_from(Feedback).where(Feedback.created_at >= since))
        ).scalar_one(),
    }


@router.get("/audit", tags=["admin"])
async def audit(session: AsyncSession = Depends(get_session), x_admin_token: str | None = Header(None)):
    """Global accounting invariants (sandbox, or with X-Admin-Token)."""
    _admin_or_sandbox(x_admin_token)
    total = (await session.execute(select(func.coalesce(func.sum(TradeLedger.amount_nanos), 0)))).scalar_one()
    unbalanced = (
        (
            await session.execute(
                select(TradeLedger.tx_id).group_by(TradeLedger.tx_id).having(func.sum(TradeLedger.amount_nanos) != 0)
            )
        )
        .scalars()
        .all()
    )
    by_account = dict(
        (
            await session.execute(select(TradeLedger.account, func.sum(TradeLedger.amount_nanos)).group_by(TradeLedger.account))
        ).all()
    )
    mismatches = []
    escrow_mismatches = []
    agents = (await session.execute(select(Agent))).scalars().all()
    for a in agents:
        if (
            by_account.get(f"agent:{a.id}:available", 0) != a.balance_available_nanos
            or by_account.get(f"agent:{a.id}:escrow", 0) != a.balance_escrow_nanos
        ):
            mismatches.append(a.id)
        order_escrow = (
            await session.execute(
                select(func.coalesce(func.sum(Order.escrow_nanos), 0)).where(Order.agent_id == a.id, Order.side == BID)
            )
        ).scalar_one()
        trade_escrow = (
            await session.execute(select(func.coalesce(func.sum(Trade.escrow_nanos), 0)).where(Trade.buyer_id == a.id))
        ).scalar_one()
        call_escrow = (
            await session.execute(
                select(func.coalesce(func.sum(ServiceCall.price_nanos), 0)).where(
                    ServiceCall.buyer_id == a.id, ServiceCall.status == "pending"
                )
            )
        ).scalar_one()
        if order_escrow + trade_escrow + call_escrow != a.balance_escrow_nanos:
            escrow_mismatches.append(a.id)
    open_asks = (
        await session.execute(select(func.count()).select_from(Order).where(Order.side == ASK, Order.status == ORDER_OPEN))
    ).scalar_one()
    return {
        "ledger_sum_nanos": int(total),
        "unbalanced_transactions": unbalanced,
        "cached_balance_mismatches": mismatches,
        "escrow_backing_mismatches": escrow_mismatches,
        "house_fees_usd": fmt_usd(int(by_account.get(ledger.HOUSE_FEES, 0))),
        "sandbox_minted_usd": fmt_usd(-int(by_account.get(ledger.HOUSE_MINT, 0))),
        "open_asks": open_asks,
        "ok": total == 0 and not unbalanced and not mismatches and not escrow_mismatches,
    }
