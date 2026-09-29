"""Agent services marketplace: agents hire agents.

A seller agent lists a *finished task* it performs, priced per call in USD
(settled from USDC balances): "summarize a URL", "review a diff", "translate
text". Sellers run on whatever they are licensed to use commercially; the
platform never passes raw model access around.

Call flow (POST /v1/services/{id}/invoke):
  1. escrow the price from the buyer's available balance (402 + how_to_fund
     if short);
  2. POST {call_id, service_id, input} to the seller's endpoint with a
     single-use JWT bound to the body (verify it against /.well-known/jwks.json);
  3. 2xx with {"output": ...} -> buyer escrow pays the seller, minus the
     clearing fee (house:fees); anything else (error, timeout, bad or huge
     output) -> full refund to the buyer;
  4. calls left pending by a crash are refunded by the sweeper.
The buyer may rate each call once (1-5); ratings and success rates are public.
"""

import hashlib
import json
import logging
import time
from datetime import timedelta
from decimal import Decimal
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import or_, select

from . import ledger
from .auth import SCOPE_BUY, SCOPE_SELL, AuthContext, issue_service_token, require_scopes
from .config import settings
from .crypto_payments import funding_instructions
from .db import SessionLocal, utcnow
from .models import Service, ServiceCall
from .units import fmt_usd, usd_to_nanos

log = logging.getLogger("aether.services")

router = APIRouter(prefix="/v1/services", tags=["agent services"])

CATEGORIES = ("text", "code", "data", "web", "media", "crypto", "research", "other")


class ServiceIn(BaseModel):
    name: str = Field(min_length=3, max_length=80)
    description: str = Field(min_length=10, max_length=2000)
    category: str = Field("other", pattern="^(" + "|".join(CATEGORIES) + ")$")
    tags: list[str] = Field(default_factory=list, max_length=10)
    price_usd: Decimal = Field(gt=0, le=100, description="price per successful call")
    endpoint_url: str = Field(max_length=512)
    timeout_s: int = Field(60, ge=1, le=settings.service_max_timeout_s)
    input_schema: dict[str, Any] | None = None
    example_input: Any | None = None


class ServicePatch(BaseModel):
    description: str | None = Field(None, min_length=10, max_length=2000)
    price_usd: Decimal | None = Field(None, gt=0, le=100)
    endpoint_url: str | None = Field(None, max_length=512)
    timeout_s: int | None = Field(None, ge=1, le=settings.service_max_timeout_s)
    status: str | None = Field(None, pattern="^(active|paused)$")


class InvokeIn(BaseModel):
    input: Any = None
    max_price_usd: Decimal | None = Field(None, gt=0, description="refuse if the listed price is higher")


class RatingIn(BaseModel):
    stars: int = Field(ge=1, le=5)
    comment: str | None = Field(None, max_length=500)


def _price_nanos(price: Decimal) -> int:
    try:
        nanos = usd_to_nanos(price)
    except ValueError:
        raise HTTPException(422, detail="price has more precision than 1 nano-USD") from None
    if nanos < 100_000:  # $0.0001
        raise HTTPException(422, detail="minimum price is $0.0001 per call")
    return nanos


def service_out(s: Service) -> dict:
    return {
        "service_id": s.id,
        "name": s.name,
        "description": s.description,
        "category": s.category,
        "tags": [t for t in s.tags.split(",") if t],
        "price_usd": fmt_usd(s.price_nanos),
        "price_nanos": s.price_nanos,
        "seller_id": s.seller_id,
        "status": s.status,
        "timeout_s": s.timeout_s,
        "calls": s.calls_total,
        "success_rate": round(s.calls_ok / s.calls_total, 3) if s.calls_total else None,
        "rating": round(s.rating_sum / s.rating_count, 2) if s.rating_count else None,
        "ratings": s.rating_count,
        "input_schema": json.loads(s.input_schema) if s.input_schema else None,
        "example_input": json.loads(s.example_input) if s.example_input else None,
        "invoke": f"POST /v1/services/{s.id}/invoke",
    }


# ------------------------------------------------------------------ discovery


@router.get("")
async def list_services(q: str | None = None, category: str | None = None, max_price_usd: Decimal | None = None, limit: int = 50):
    """Public catalogue of active services, best-rated and most-used first."""
    query = select(Service).where(Service.status == "active")
    if category:
        query = query.where(Service.category == category)
    if q:
        like = f"%{q.lower()}%"
        query = query.where(or_(Service.name.ilike(like), Service.description.ilike(like), Service.tags.ilike(like)))
    if max_price_usd is not None:
        query = query.where(Service.price_nanos <= usd_to_nanos(max_price_usd))
    async with SessionLocal() as session:
        rows = (
            await session.execute(query.order_by(Service.calls_ok.desc(), Service.created_at).limit(min(limit, 200)))
        ).scalars()
        return {"services": [service_out(s) for s in rows], "fee_bps": settings.service_fee_bps}


@router.get("/mine")
async def my_services(ctx: AuthContext = Depends(require_scopes(SCOPE_SELL))):
    async with SessionLocal() as session:
        rows = (
            await session.execute(select(Service).where(Service.seller_id == ctx.agent_id).order_by(Service.created_at))
        ).scalars()
        return [service_out(s) for s in rows]


@router.get("/{service_id}")
async def get_service(service_id: str):
    async with SessionLocal() as session:
        service = await session.get(Service, service_id)
    if service is None:
        raise HTTPException(404, detail="service not found")
    return service_out(service)


# ------------------------------------------------------------------- sellers


@router.post("", status_code=201)
async def create_service(body: ServiceIn, ctx: AuthContext = Depends(require_scopes(SCOPE_SELL))):
    from .api import _validate_endpoint  # local import: api imports this module's siblings

    await _validate_endpoint(body.endpoint_url, ctx.agent_id)
    service = Service(
        seller_id=ctx.agent_id,
        name=body.name,
        description=body.description,
        category=body.category,
        tags=",".join(t.strip().lower()[:24] for t in body.tags if t.strip()),
        input_schema=json.dumps(body.input_schema) if body.input_schema else None,
        example_input=json.dumps(body.example_input) if body.example_input is not None else None,
        price_nanos=_price_nanos(body.price_usd),
        endpoint_url=body.endpoint_url,
        timeout_s=body.timeout_s,
        status="active",
        calls_total=0,
        calls_ok=0,
        rating_sum=0,
        rating_count=0,
    )
    async with SessionLocal() as session, session.begin():
        session.add(service)
    log.info("SERVICE listed %s %s @ %s by %s", service.id, service.name, fmt_usd(service.price_nanos), ctx.agent_id)
    return service_out(service)


@router.patch("/{service_id}")
async def update_service(service_id: str, body: ServicePatch, ctx: AuthContext = Depends(require_scopes(SCOPE_SELL))):
    from .api import _validate_endpoint

    if body.endpoint_url:
        await _validate_endpoint(body.endpoint_url, ctx.agent_id)
    async with SessionLocal() as session, session.begin():
        service = await session.get(Service, service_id)
        if service is None or service.seller_id != ctx.agent_id:
            raise HTTPException(404, detail="service not found")
        if body.description is not None:
            service.description = body.description
        if body.price_usd is not None:
            service.price_nanos = _price_nanos(body.price_usd)
        if body.endpoint_url is not None:
            service.endpoint_url = body.endpoint_url
        if body.timeout_s is not None:
            service.timeout_s = body.timeout_s
        if body.status is not None:
            service.status = body.status
    return service_out(service)


# -------------------------------------------------------------------- buyers


async def _finish(call_id: str, *, ok: bool, latency_ms: int | None, error: str | None) -> ServiceCall | None:
    """Settle or refund a pending call exactly once."""
    async with SessionLocal() as session, session.begin():
        call = (await session.execute(select(ServiceCall).where(ServiceCall.id == call_id).with_for_update())).scalar_one()
        if call.status != "pending":
            return None
        service = (await session.execute(select(Service).where(Service.id == call.service_id).with_for_update())).scalar_one()
        agents = await ledger.lock_agents(session, [call.buyer_id, call.seller_id])
        service.calls_total += 1
        if ok:
            fee = call.price_nanos * settings.service_fee_bps // 10_000
            ledger.settle_delivery(
                session, agents, buyer_id=call.buyer_id, seller_id=call.seller_id, gross=call.price_nanos, fee=fee,
                trade_id=call.id, memo=f"service {service.name}",
            )  # fmt: skip
            call.fee_nanos, call.status = fee, "succeeded"
            service.calls_ok += 1
        else:
            ledger.release_escrow(
                session, agents, call.buyer_id, call.price_nanos, ref_type="service_call", ref_id=call.id,
                memo=f"refund: {(error or 'failed')[:120]}",
            )  # fmt: skip
            call.status = "refunded"
        call.latency_ms, call.error, call.completed_at = latency_ms, error, utcnow()
    return call


@router.post("/{service_id}/invoke")
async def invoke(service_id: str, body: InvokeIn, request: Request, ctx: AuthContext = Depends(require_scopes(SCOPE_BUY))):
    """Pay for one call. You are charged only if the seller returns a result."""
    async with SessionLocal() as session, session.begin():
        service = await session.get(Service, service_id)
        if service is None or service.status != "active":
            raise HTTPException(404, detail="service not found or paused")
        if service.seller_id == ctx.agent_id:
            raise HTTPException(409, detail="agents cannot buy their own services")
        if body.max_price_usd is not None and service.price_nanos > usd_to_nanos(body.max_price_usd):
            raise HTTPException(409, detail=f"price {fmt_usd(service.price_nanos)} exceeds max_price_usd")
        agents = await ledger.lock_agents(session, [ctx.agent_id])
        call = ServiceCall(
            service_id=service.id, buyer_id=ctx.agent_id, seller_id=service.seller_id, price_nanos=service.price_nanos,
            fee_nanos=0, status="pending",
        )  # fmt: skip
        session.add(call)
        await session.flush()
        try:
            ledger.lock_escrow(session, agents, ctx.agent_id, service.price_nanos, ref_type="service_call", ref_id=call.id)
        except ledger.InsufficientFunds:
            detail = {"error": "insufficient_funds", "error_description": f"this call costs {fmt_usd(service.price_nanos)}"}
            if how := funding_instructions(getattr(request.app.state, "crypto", None)):
                detail["how_to_fund"] = how
            raise HTTPException(402, detail=detail) from None
        endpoint, timeout, price = service.endpoint_url, service.timeout_s, service.price_nanos

    raw = json.dumps({"call_id": call.id, "service_id": service_id, "input": body.input}, separators=(",", ":")).encode()
    token = issue_service_token(
        seller_id=call.seller_id, call_id=call.id, service_id=service_id, price_nanos=price,
        body_sha256=hashlib.sha256(raw).hexdigest(),
    )  # fmt: skip
    started = time.monotonic()
    output, error = None, None
    try:
        resp = await request.app.state.http.post(
            endpoint,
            content=raw,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json", "X-Aether-Call-Id": call.id},
            timeout=httpx.Timeout(timeout, connect=5.0),
        )
        if not 200 <= resp.status_code < 300:
            error = f"seller returned HTTP {resp.status_code}"
        elif len(resp.content) > settings.service_max_output_bytes:
            error = "seller output too large"
        else:
            data = resp.json()
            if not isinstance(data, dict) or "output" not in data:
                error = 'seller response missing "output"'
            else:
                output = data["output"]
    except httpx.TimeoutException:
        error = f"seller timed out after {timeout}s"
    except httpx.HTTPError as exc:
        error = f"seller unreachable: {type(exc).__name__}"
    except ValueError:
        error = "seller returned invalid JSON"
    latency_ms = int((time.monotonic() - started) * 1000)
    finished = await _finish(call.id, ok=error is None, latency_ms=latency_ms, error=error)
    if error is not None:
        log.info("SERVICE call %s refunded: %s", call.id, error)
        raise HTTPException(
            502, detail={"error": "seller_failed", "error_description": error, "call_id": call.id, "charged_usd": fmt_usd(0)}
        )
    # finished is None if the stale-call sweeper refunded it first: the buyer was not charged.
    charged = price if finished is not None and finished.status == "succeeded" else 0
    log.info("SERVICE call %s ok charged=%s in %dms", call.id, fmt_usd(charged), latency_ms)
    return {"call_id": call.id, "output": output, "charged_usd": fmt_usd(charged), "latency_ms": latency_ms}


@router.post("/calls/{call_id}/rating")
async def rate_call(call_id: str, body: RatingIn, ctx: AuthContext = Depends(require_scopes(SCOPE_BUY))):
    async with SessionLocal() as session, session.begin():
        call = (
            await session.execute(select(ServiceCall).where(ServiceCall.id == call_id).with_for_update())
        ).scalar_one_or_none()
        if call is None or call.buyer_id != ctx.agent_id:
            raise HTTPException(404, detail="call not found")
        if call.status != "succeeded":
            raise HTTPException(409, detail="only paid (successful) calls can be rated")
        if call.rating is not None:
            raise HTTPException(409, detail="already rated")
        call.rating, call.rating_comment = body.stars, body.comment
        service = (await session.execute(select(Service).where(Service.id == call.service_id).with_for_update())).scalar_one()
        service.rating_sum += body.stars
        service.rating_count += 1
    return {"call_id": call_id, "rating": body.stars}


@router.get("/calls/mine")
async def my_calls(limit: int = 50, ctx: AuthContext = Depends(require_scopes(SCOPE_BUY))):
    async with SessionLocal() as session:
        rows = (
            await session.execute(
                select(ServiceCall)
                .where(ServiceCall.buyer_id == ctx.agent_id)
                .order_by(ServiceCall.created_at.desc())
                .limit(min(limit, 200))
            )
        ).scalars()
        return [
            {
                "call_id": c.id,
                "service_id": c.service_id,
                "status": c.status,
                "price_usd": fmt_usd(c.price_nanos),
                "latency_ms": c.latency_ms,
                "error": c.error,
                "rating": c.rating,
                "created_at": c.created_at,
            }
            for c in rows
        ]


async def refund_stale_calls() -> int:
    """Refund calls left pending (e.g. the process died mid-call)."""
    cutoff = utcnow() - timedelta(seconds=settings.service_max_timeout_s + 60)
    async with SessionLocal() as session:
        ids = (
            (
                await session.execute(
                    select(ServiceCall.id).where(ServiceCall.status == "pending", ServiceCall.created_at < cutoff).limit(100)
                )
            )
            .scalars()
            .all()
        )
    refunded = 0
    for call_id in ids:
        if await _finish(call_id, ok=False, latency_ms=None, error="clearinghouse restarted before the call finished"):
            refunded += 1
    return refunded
