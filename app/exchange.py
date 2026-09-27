"""Clearing & settlement: turns matching-engine events into escrowed
allocations, and inference deliveries into seller payouts.

Money flow for one unit of inference:

  1. bid placed      buyer available -> buyer escrow        (price_cap * qty)
  2. fill            escrow moves from the bid order to a Trade (allocation);
                     any price improvement goes back to available
  3. proxy request   tokens reserved on the Trade (no money moves yet)
  4. delivery        buyer escrow -> seller available + house fee,
                     for exactly the tokens the proxy streamed
  5. release/expiry  unused allocation escrow -> buyer available

Lock order everywhere: orders / trades (sorted by id) first, agents last.
"""

import asyncio
import json
import logging
import os
import socket
import time
from dataclasses import asdict, dataclass
from datetime import timedelta

from redis.asyncio import Redis
from redis.exceptions import ResponseError
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from . import ledger
from .config import settings
from .db import SessionLocal, aware, utcnow
from .matching_engine import MatchEvent, MatchingEngine
from .models import (
    ASK,
    BID,
    GTC,
    IOC,
    JOB_STREAMING,
    MARKET,
    ORDER_CANCELLED,
    ORDER_EXPIRED,
    ORDER_FILLED,
    ORDER_OPEN,
    TRADE_ACTIVE,
    TRADE_EXHAUSTED,
    TRADE_EXPIRED,
    TRADE_RELEASED,
    Agent,
    AppliedMatchEvent,
    InferenceJob,
    Order,
    Trade,
    new_id,
)
from .units import npt_to_usd_per_mtok

log = logging.getLogger("aether.exchange")


class ExchangeError(Exception):
    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.message = message


# ------------------------------------------------------------------ order state


def _refresh_status(order: Order) -> None:
    if order.status in (ORDER_CANCELLED, ORDER_EXPIRED):
        return  # terminal states set explicitly; late fills still count in filled_tokens
    order.status = ORDER_FILLED if order.open_tokens == 0 else ORDER_OPEN


def _retire(session, agents, order: Order, tokens: int, status: str, memo: str) -> None:
    """Remove `tokens` from an order's open quantity and refund bid escrow."""
    if tokens > 0:
        order.cancelled_tokens += tokens
        if order.side == BID:
            amount = order.price_npt * tokens
            order.escrow_nanos -= amount
            ledger.release_escrow(session, agents, order.agent_id, amount, ref_type="order", ref_id=order.id, memo=memo)
        order.status = status
    _refresh_status(order)


# --------------------------------------------------------------- order entry


async def place_order(
    engine: MatchingEngine,
    redis: Redis,
    *,
    agent_id: str,
    instrument: str,
    side: str,
    order_type: str,
    time_in_force: str,
    price_npt: int,
    quantity: int,
    ttl_s: int | None,
) -> tuple[Order, list[Trade]]:
    if order_type == MARKET:
        time_in_force = IOC  # market orders never rest
    if side == ASK and time_in_force == GTC and ttl_s is None:
        ttl_s = settings.default_ask_ttl_s  # spot capacity is perishable
    expires_at = utcnow() + timedelta(seconds=ttl_s) if ttl_s else None
    seq = await engine.next_seq()

    async with SessionLocal() as session, session.begin():
        agents = await ledger.lock_agents(session, [agent_id])
        if side == ASK and not agents[agent_id].endpoint_url:
            raise ExchangeError(409, "register an inference endpoint (PUT /v1/agents/me/endpoint) before selling")
        order = Order(
            id=new_id("ord"),
            agent_id=agent_id,
            instrument=instrument,
            side=side,
            order_type=order_type,
            time_in_force=time_in_force,
            price_npt=price_npt,
            quantity_tokens=quantity,
            filled_tokens=0,
            cancelled_tokens=0,
            escrow_nanos=0,
            status=ORDER_OPEN,
            seq=seq,
            expires_at=expires_at,
        )
        session.add(order)
        if side == BID:
            # Escrow first: nothing reaches the book without fully funded collateral.
            escrow = price_npt * quantity
            try:
                ledger.lock_escrow(session, agents, agent_id, escrow, ref_type="order", ref_id=order.id)
            except ledger.InsufficientFunds:
                raise ExchangeError(402, f"insufficient available balance to escrow {escrow} nano-USD for this bid") from None
            order.escrow_nanos = escrow

    event = await engine.submit(order, rest=(time_in_force == GTC))
    await apply_match_event(event, redis)

    async with SessionLocal() as session:
        order = await session.get(Order, order.id)
        trades = (await session.execute(select(Trade).where(Trade.event_id == event.event_id).order_by(Trade.id))).scalars().all()
    return order, list(trades)


async def cancel_order(engine: MatchingEngine, *, agent_id: str, order_id: str) -> Order:
    async with SessionLocal() as session:
        order = await session.get(Order, order_id)
    if order is None or order.agent_id != agent_id:
        raise ExchangeError(404, "order not found")
    if order.status != ORDER_OPEN:
        raise ExchangeError(409, f"order is already {order.status}")

    remaining = await engine.cancel(order_id)
    async with SessionLocal() as session, session.begin():
        order = (await session.execute(select(Order).where(Order.id == order_id).with_for_update())).scalar_one()
        if remaining is None:
            raise ExchangeError(409, f"order is not resting on the book (status {order.status})")
        agents = await ledger.lock_agents(session, [agent_id]) if order.side == BID else {}
        _retire(session, agents, order, remaining, ORDER_CANCELLED, memo="cancelled by agent")
    return order


# --------------------------------------------------------- match application


async def apply_match_event(event: MatchEvent, redis: Redis | None = None) -> list[Trade]:
    """Apply one matching-engine event to PostgreSQL exactly once."""
    order_ids = {f.maker_order_id for f in event.fills} | {oid for oid, _ in event.expired}
    if event.taker_order_id:
        order_ids.add(event.taker_order_id)
    trades: list[Trade] = []
    try:
        async with SessionLocal() as session, session.begin():
            if await session.get(AppliedMatchEvent, event.event_id) is not None:
                return []
            session.add(AppliedMatchEvent(event_id=event.event_id))
            await session.flush()  # claims the event; a concurrent applier blocks, then conflicts

            rows = await session.execute(
                select(Order).where(Order.id.in_(sorted(order_ids))).order_by(Order.id).with_for_update()
            )
            orders = {o.id: o for o in rows.scalars()}
            agents = await ledger.lock_agents(session, {o.agent_id for o in orders.values() if o.side == BID})
            taker = orders.get(event.taker_order_id)
            now = utcnow()

            for n, fill in enumerate(event.fills):
                maker = orders.get(fill.maker_order_id)
                if maker is None or taker is None:
                    log.error("event %s references unknown order(s); skipping fill %d", event.event_id, n)
                    continue
                bid, ask = (taker, maker) if taker.side == BID else (maker, taker)
                q, p = fill.quantity, fill.price_npt  # executes at the maker's price
                expires_at = now + timedelta(seconds=settings.allocation_ttl_s)
                if ask.expires_at is not None:
                    expires_at = min(expires_at, aware(ask.expires_at))
                trade = Trade(
                    id=f"trd_{event.event_id.replace('-', '_')}_{n}",
                    event_id=event.event_id,
                    instrument=bid.instrument,
                    bid_order_id=bid.id,
                    ask_order_id=ask.id,
                    buyer_id=bid.agent_id,
                    seller_id=ask.agent_id,
                    taker_side=taker.side,
                    price_npt=p,
                    tokens_total=q,
                    tokens_used=0,
                    tokens_reserved=0,
                    escrow_nanos=p * q,
                    gross_settled_nanos=0,
                    fee_nanos=0,
                    status=TRADE_ACTIVE,
                    expires_at=expires_at,
                )
                session.add(trade)
                bid.escrow_nanos -= bid.price_npt * q
                improvement = (bid.price_npt - p) * q
                if improvement:
                    ledger.release_escrow(
                        session,
                        agents,
                        bid.agent_id,
                        improvement,
                        ref_type="trade",
                        ref_id=trade.id,
                        memo="price improvement",
                    )
                bid.filled_tokens += q
                ask.filled_tokens += q
                _refresh_status(bid)
                _refresh_status(ask)
                trades.append(trade)

            for oid, remaining in event.expired:
                if oid in orders:
                    _retire(session, agents, orders[oid], remaining, ORDER_EXPIRED, memo="order expired")

            if taker is not None and event.type == "match" and not event.rested:
                _retire(session, agents, taker, event.remaining, ORDER_CANCELLED, memo="unfilled IOC remainder")
    except IntegrityError:
        log.debug("event %s already applied concurrently", event.event_id)
        return []

    if redis is not None and trades:
        await _publish_trades(redis, trades)
    for t in trades:
        log.info(
            "FILL %s %s tokens @ $%s/1M buyer=%s seller=%s (taker=%s)",
            t.instrument,
            t.tokens_total,
            npt_to_usd_per_mtok(t.price_npt),
            t.buyer_id,
            t.seller_id,
            t.taker_side,
        )
    return trades


async def _publish_trades(redis: Redis, trades: list[Trade]) -> None:
    pipe = redis.pipeline(transaction=False)
    for t in trades:
        msg = json.dumps(
            {
                "type": "trade",
                "trade_id": t.id,
                "instrument": t.instrument,
                "price_usd_per_mtok": npt_to_usd_per_mtok(t.price_npt),
                "tokens": t.tokens_total,
                "taker_side": t.taker_side,
                "ts": time.time(),
            }
        )
        pipe.publish(f"{settings.redis_prefix}:md:{t.instrument}", msg)
    await pipe.execute()


# ------------------------------------------------------- allocation lifecycle


async def release_trade(*, agent_id: str, trade_id: str) -> Trade:
    async with SessionLocal() as session, session.begin():
        trade = (await session.execute(select(Trade).where(Trade.id == trade_id).with_for_update())).scalar_one_or_none()
        if trade is None or trade.buyer_id != agent_id:
            raise ExchangeError(404, "allocation not found")
        if trade.status != TRADE_ACTIVE:
            raise ExchangeError(409, f"allocation is already {trade.status}")
        if trade.tokens_reserved:
            raise ExchangeError(409, "allocation has in-flight inference requests")
        agents = await ledger.lock_agents(session, [trade.buyer_id])
        _close_trade(session, agents, trade, TRADE_RELEASED, "allocation released by buyer")
    return trade


def _close_trade(session, agents, trade: Trade, status: str, memo: str) -> None:
    ledger.release_escrow(session, agents, trade.buyer_id, trade.escrow_nanos, ref_type="trade", ref_id=trade.id, memo=memo)
    trade.escrow_nanos = 0
    trade.status = status


async def expire_allocations(limit: int = 200) -> int:
    now = utcnow()
    async with SessionLocal() as session:
        ids = (
            (
                await session.execute(
                    select(Trade.id)
                    .where(Trade.status == TRADE_ACTIVE, Trade.expires_at <= now, Trade.tokens_reserved == 0)
                    .limit(limit)
                )
            )
            .scalars()
            .all()
        )
    expired = 0
    for trade_id in ids:
        async with SessionLocal() as session, session.begin():
            trade = (await session.execute(select(Trade).where(Trade.id == trade_id).with_for_update())).scalar_one()
            if trade.status != TRADE_ACTIVE or trade.tokens_reserved:
                continue
            agents = await ledger.lock_agents(session, [trade.buyer_id])
            _close_trade(session, agents, trade, TRADE_EXPIRED, "allocation expired unused")
            expired += 1
    return expired


# ------------------------------------------------------ inference settlement


@dataclass
class Segment:
    """One seller's contribution to an inference job."""

    trade_id: str
    seller_id: str
    endpoint_url: str
    price_npt: int
    reserved: int
    delivered: int = 0


async def reserve_allocation(
    *, buyer_id: str, instrument: str, tokens: int, trade_id: str | None = None, exclude: set[str] | frozenset = frozenset()
) -> Segment | None:
    """Pick the cheapest escrowed allocation that can cover `tokens` and reserve
    them, so concurrent requests can never overdraw the same allocation."""
    now = utcnow()
    async with SessionLocal() as session, session.begin():
        query = (
            select(Trade, Agent.endpoint_url)
            .join(Agent, Agent.id == Trade.seller_id)
            .where(
                Trade.buyer_id == buyer_id,
                Trade.instrument == instrument,
                Trade.status == TRADE_ACTIVE,
                Trade.expires_at > now,
                Trade.tokens_total - Trade.tokens_used - Trade.tokens_reserved >= tokens,
                Agent.is_active.is_(True),
                Agent.endpoint_url.is_not(None),
            )
            .order_by(Trade.price_npt, Trade.created_at)
            .limit(1)
            # Plain FOR UPDATE (not SKIP LOCKED): under contention we wait a few
            # microseconds rather than routing to a pricier allocation.
            .with_for_update(of=Trade)
        )
        if trade_id is not None:
            query = query.where(Trade.id == trade_id)
        if exclude:
            query = query.where(Trade.id.not_in(sorted(exclude)))
        row = (await session.execute(query)).first()
        if row is None:
            return None
        trade, endpoint_url = row
        if trade.escrow_nanos < trade.price_npt * (trade.tokens_total - trade.tokens_used):
            log.error("allocation %s is under-escrowed; refusing to route", trade.id)
            return None
        trade.tokens_reserved += tokens
        return Segment(trade.id, trade.seller_id, endpoint_url, trade.price_npt, tokens)


async def settle_job(
    *,
    job_id: str,
    buyer_id: str,
    segments: list[Segment],
    status: str,
    finish_reason: str | None,
    attempts: int,
    failovers: int,
    error: str | None,
) -> int | None:
    """Release reservations, pay sellers for delivered tokens, finalise the job.
    Returns the gross cost charged to the buyer in nano-USD, or None if the job
    was already settled. Idempotent across processes via the job row lock."""
    total = 0
    async with SessionLocal() as session, session.begin():
        job = (
            await session.execute(select(InferenceJob).where(InferenceJob.id == job_id).with_for_update())
        ).scalar_one_or_none()
        if job is None or job.status != JOB_STREAMING:
            return None  # already settled (e.g. by the orphan-recovery sweep)
        rows = await session.execute(
            select(Trade).where(Trade.id.in_(sorted({s.trade_id for s in segments}))).order_by(Trade.id).with_for_update()
        )
        trades = {t.id: t for t in rows.scalars()}
        agents = await ledger.lock_agents(session, {buyer_id} | {s.seller_id for s in segments})
        for seg in segments:
            trade = trades[seg.trade_id]
            delivered = min(seg.delivered, seg.reserved)
            trade.tokens_reserved -= seg.reserved
            if delivered:
                gross = trade.price_npt * delivered
                gross_total = trade.gross_settled_nanos + gross
                fee_total = gross_total * settings.fee_bps // 10_000  # cumulative => no drift
                fee = fee_total - trade.fee_nanos
                trade.tokens_used += delivered
                trade.escrow_nanos -= gross
                trade.gross_settled_nanos, trade.fee_nanos = gross_total, fee_total
                ledger.settle_delivery(
                    session,
                    agents,
                    buyer_id=buyer_id,
                    seller_id=seg.seller_id,
                    gross=gross,
                    fee=fee,
                    trade_id=trade.id,
                    memo=f"{job_id}: {delivered} tokens",
                )
                total += gross
            if trade.tokens_used == trade.tokens_total and trade.status == TRADE_ACTIVE:
                trade.status = TRADE_EXHAUSTED
        job.status = status
        job.finish_reason = finish_reason
        job.tokens_delivered = sum(min(s.delivered, s.reserved) for s in segments)
        job.cost_nanos = total
        job.attempts = attempts
        job.failovers = failovers
        job.error = error
        job.segments = json.dumps([{k: v for k, v in asdict(s).items() if k != "endpoint_url"} for s in segments])
        job.completed_at = utcnow()
    return total


# --------------------------------------------------------- background workers


class SettlementWorker:
    """Consumer-group reader of the match stream. The request path already
    applies its own events inline; this worker guarantees that events survive a
    crash between the Lua match and the PostgreSQL commit."""

    GROUP = "settlement"

    def __init__(self, redis: Redis, engine: MatchingEngine, consumer: str | None = None):
        self.redis = redis
        self.stream = engine.stream_key
        self.consumer = consumer or f"{socket.gethostname()}-{os.getpid()}"

    async def ensure_group(self) -> None:
        try:
            await self.redis.xgroup_create(self.stream, self.GROUP, id="0", mkstream=True)
        except ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    async def _handle(self, entries) -> None:
        for entry_id, fields in entries:
            if not fields:  # trimmed from the stream
                await self.redis.xack(self.stream, self.GROUP, entry_id)
                continue
            event = MatchEvent.from_payload(entry_id, fields.get("type", "match"), fields["payload"])
            await apply_match_event(event, self.redis)
            await self.redis.xack(self.stream, self.GROUP, entry_id)

    async def _claim_stale(self, min_idle_ms: int) -> None:
        start = "0-0"
        while True:
            result = await self.redis.xautoclaim(
                self.stream, self.GROUP, self.consumer, min_idle_time=min_idle_ms, start_id=start, count=100
            )
            await self._handle(result[1])
            start = result[0]
            if start in ("0-0", b"0-0"):
                return

    async def drain(self) -> None:
        """Apply every unacknowledged event (used at startup before reconcile)."""
        await self.ensure_group()
        await self._claim_stale(0)  # idempotent apply makes re-processing safe
        while True:
            resp = await self.redis.xreadgroup(self.GROUP, self.consumer, {self.stream: ">"}, count=200)
            if not resp:
                return
            for _, entries in resp:
                await self._handle(entries)

    async def run(self) -> None:
        await self.ensure_group()
        last_claim = time.monotonic()
        while True:
            try:
                resp = await self.redis.xreadgroup(self.GROUP, self.consumer, {self.stream: ">"}, count=200, block=2000)
                for _, entries in resp or []:
                    await self._handle(entries)
                if time.monotonic() - last_claim > 30:
                    await self._claim_stale(30_000)
                    last_claim = time.monotonic()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("settlement worker error")
                await asyncio.sleep(1)


async def reconcile_book(engine: MatchingEngine) -> dict:
    """PostgreSQL is authoritative. After the stream is drained: re-insert open
    GTC orders Redis lost, and close IOC orders that never reached the engine."""
    restored = closed = 0
    async with SessionLocal() as session:
        open_orders = (await session.execute(select(Order).where(Order.status == ORDER_OPEN))).scalars().all()
        max_seq = (await session.execute(select(func.max(Order.seq)))).scalar_one() or 0
    # Keep time priority monotonic even if Redis lost the sequence counter.
    if int(await engine.redis.get(engine.seq_key) or 0) < max_seq:
        await engine.redis.set(engine.seq_key, max_seq)
    for order in open_orders:
        if await engine.is_resting(order.id):
            continue
        if order.time_in_force == GTC and (order.expires_at is None or aware(order.expires_at) > utcnow()):
            await engine.restore(order)
            restored += 1
            continue
        async with SessionLocal() as session, session.begin():
            locked = (await session.execute(select(Order).where(Order.id == order.id).with_for_update())).scalar_one()
            agents = await ledger.lock_agents(session, [locked.agent_id]) if locked.side == BID else {}
            status = ORDER_EXPIRED if locked.time_in_force == GTC else ORDER_CANCELLED
            _retire(session, agents, locked, locked.open_tokens, status, memo="reconciled at startup")
            closed += 1
    if restored or closed:
        log.warning("book reconcile: restored=%d closed=%d", restored, closed)
    return {"restored": restored, "closed": closed}


async def run_sweeps(engine: MatchingEngine, redis: Redis) -> None:
    """One maintenance pass: expire resting orders and unused allocations."""
    event = await engine.sweep_expired()
    if event is not None:
        await apply_match_event(event, redis)
    await expire_allocations()


__all__ = [
    "ExchangeError",
    "Segment",
    "SettlementWorker",
    "apply_match_event",
    "cancel_order",
    "expire_allocations",
    "place_order",
    "reconcile_book",
    "release_trade",
    "reserve_allocation",
    "settle_job",
    "run_sweeps",
]
