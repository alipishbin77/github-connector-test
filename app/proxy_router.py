"""The authenticated inference proxy — the only path by which inference units
are consumed.

Implementation choice: Python asyncio + a shared pooled httpx.AsyncClient in
the same process as the API. The router is I/O-bound (it relays chunks and
does a few small DB transactions per request), so one event loop multiplexes
thousands of concurrent streams; keeping it in Python lets it share the
ledger/matching code with no RPC hop. A Go data plane is the obvious next
step once per-chunk CPU (tokenisation, metering) dominates.

Request lifecycle:
  1. Verify the buyer's credential (scope buy_inference).
  2. Escrow gate: atomically reserve max_tokens across the buyer's escrowed
     allocations, cheapest first. If they cannot cover it -> 402 (or, for the
     OpenAI-compatible endpoint, auto-buy the deficit first). Nothing is routed
     against unfunded capacity.
  3. For each planned segment, mint a single-use delivery JWT bound to the
     seller, job and body hash; POST to the seller's endpoint with an absolute
     token bound for that segment; relay its SSE stream.
  4. Checkpoint delivered tokens to a Redis Stream. When a segment's capacity
     is used up, roll over to the next allocation. If the seller's instance is
     interrupted (drop, timeout, 5xx, or a stream that ends without a
     completion marker), resume from the checkpoint on the same allocation,
     then fail over to replacement capacity. The buyer sees one continuous
     stream with contiguous token indices.
  5. Settle: pay each seller for exactly the tokens it delivered and release
     the rest of every reservation. Runs even if the buyer disconnects.
"""

import asyncio
import hashlib
import json
import logging
import time
from collections.abc import Awaitable, Callable
from contextlib import aclosing
from datetime import timedelta

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from redis.asyncio import Redis
from sqlalchemy import select

from .auth import SCOPE_BUY, AuthContext, issue_delivery_token, require_scopes
from .config import settings
from .db import SessionLocal, utcnow
from .exchange import Segment, reserve_capacity, settle_job
from .models import JOB_COMPLETED, JOB_FAILED, JOB_REJECTED, JOB_STREAMING, InferenceJob, new_id
from .units import fmt_usd, npt_to_usd_per_mtok

log = logging.getLogger("aether.proxy")

router = APIRouter(tags=["inference"])

# Strong references to settlement tasks spawned while a client disconnects.
_background: set[asyncio.Task] = set()

INSTRUMENT_PATTERN = r"^[a-z0-9][a-z0-9._:-]{1,63}$"

# Called with the number of tokens still needed; returns replacement segments.
CapacitySource = Callable[[int, set[str]], Awaitable[list[Segment] | None]]


class InferenceRequest(BaseModel):
    instrument: str = Field(pattern=INSTRUMENT_PATTERN)
    prompt: str = Field(min_length=1, max_length=200_000)
    max_tokens: int = Field(gt=0, le=settings.max_tokens_per_request)
    trade_id: str | None = Field(None, description="pin a specific allocation; default is cheapest")


class SellerInterrupted(Exception):
    """Transient: spot preemption, network drop, timeout, 5xx. Retry, then fail over."""


class SellerRejected(Exception):
    """Permanent for this seller (4xx, protocol violation). Fail over immediately."""


def sse(data: dict, event: str | None = None) -> bytes:
    head = f"event: {event}\n" if event else ""
    return f"{head}data: {json.dumps(data, separators=(',', ':'))}\n\n".encode()


# ------------------------------------------------------------------ checkpoints


class Checkpointer:
    """Stateless-retry support: every delivered token is durably appended (in
    batches) to a Redis Stream, together with which segment produced it. A
    resumed request needs only (resume_from, prefix), so any seller replica
    can continue the job. The same data lets another proxy process settle a
    job whose proxy died mid-stream (see recover_orphaned_jobs)."""

    def __init__(self, redis: Redis, job_id: str):
        p = f"{settings.redis_prefix}:job:{job_id}"
        self.redis = redis
        self.stream_key = f"{p}:ckpt"
        self.segments_key = f"{p}:segments"
        self.heartbeat_key = f"{p}:hb"
        self._buf: list[list] = []
        self._last_beat = 0.0

    async def save_segments(self, segments: list[Segment]) -> None:
        data = json.dumps([{k: v for k, v in s.__dict__.items() if k != "delivered"} for s in segments])
        pipe = self.redis.pipeline(transaction=False)
        pipe.set(self.segments_key, data, ex=settings.checkpoint_ttl_s)
        pipe.set(self.heartbeat_key, "1", ex=120)
        await pipe.execute()

    async def add(self, index: int, token: str, segment: int) -> None:
        self._buf.append([index, token, segment])
        if len(self._buf) >= settings.checkpoint_every_tokens or time.monotonic() - self._last_beat > 10:
            await self.flush()

    async def flush(self) -> None:
        pipe = self.redis.pipeline(transaction=False)
        if self._buf:
            pipe.xadd(self.stream_key, {"tokens": json.dumps(self._buf, separators=(",", ":"))})
            pipe.expire(self.stream_key, settings.checkpoint_ttl_s)
        pipe.set(self.heartbeat_key, "1", ex=120)
        await pipe.execute()
        self._buf.clear()
        self._last_beat = time.monotonic()

    async def finish(self) -> None:
        await self.flush()
        await self.redis.delete(self.heartbeat_key)


async def read_checkpoint(redis: Redis, job_id: str) -> list[list]:
    entries = await redis.xrange(f"{settings.redis_prefix}:job:{job_id}:ckpt")
    tokens: list[list] = []
    for _, fields in entries:
        tokens.extend(json.loads(fields["tokens"]))
    return tokens


# ------------------------------------------------------------------- the router


class InferenceRun:
    """One metered inference job. `events()` yields (kind, data) tuples —
    meta, token, resume, done|error — that the native and OpenAI-compatible
    endpoints format for their clients."""

    def __init__(
        self,
        http: httpx.AsyncClient,
        redis: Redis,
        buyer_id: str,
        job_id: str,
        *,
        instrument: str,
        prompt: str,
        max_tokens: int,
        plan: list[Segment],
        messages: list[dict] | None = None,
        more_capacity: CapacitySource | None = None,
    ):
        self.http = http
        self.redis = redis
        self.buyer_id = buyer_id
        self.job_id = job_id
        self.instrument = instrument
        self.prompt = prompt
        self.messages = messages
        self.max_tokens = max_tokens
        self.segments: list[Segment] = list(plan)  # everything reserved (settled at the end)
        self.queue: list[int] = list(range(len(plan)))  # indices of segments still to consume
        self.current = self.queue.pop(0)
        self.failed_trades: set[str] = set()
        self.more_capacity = more_capacity
        self.parts: list[str] = []
        self.delivered = 0
        self.attempts = 0
        self.failovers = 0
        self.finish_reason: str | None = None
        self.error: str | None = None
        self.cost: int | None = None
        self.ckpt = Checkpointer(redis, job_id)
        self._finalize_lock = asyncio.Lock()
        self._finalized = False

    @property
    def seg(self) -> Segment:
        return self.segments[self.current]

    # -- seller call -------------------------------------------------------------
    async def _call_seller(self, seg: Segment, end: int):
        """Yields (index, token, None) per token and finally (None, None, finish_reason)."""
        body = {
            "job_id": self.job_id,
            "instrument": self.instrument,
            "prompt": self.prompt,
            "max_tokens": end,  # absolute index bound for this call
            "resume_from": self.delivered,
            "prefix": "".join(self.parts),
        }
        if self.messages is not None:
            body["messages"] = self.messages
        raw = json.dumps(body, separators=(",", ":")).encode()
        token = issue_delivery_token(
            seller_id=seg.seller_id,
            job_id=self.job_id,
            trade_id=seg.trade_id,
            instrument=self.instrument,
            max_tokens=end,
            resume_from=self.delivered,
            body_sha256=hashlib.sha256(raw).hexdigest(),
        )
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "X-Aether-Job-Id": self.job_id,
        }
        timeout = httpx.Timeout(
            connect=settings.seller_connect_timeout_s, read=settings.seller_read_timeout_s, write=5.0, pool=5.0
        )
        try:
            async with self.http.stream("POST", seg.endpoint_url, content=raw, headers=headers, timeout=timeout) as resp:
                if resp.status_code >= 500 or resp.status_code == 429:
                    raise SellerInterrupted(f"seller returned HTTP {resp.status_code}")
                if resp.status_code != 200:
                    detail = (await resp.aread())[:200].decode(errors="replace")
                    raise SellerRejected(f"seller returned HTTP {resp.status_code}: {detail}")
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = json.loads(line[5:].strip())
                    if data.get("done"):
                        yield None, None, str(data.get("finish_reason") or "stop")
                        return
                    yield int(data["index"]), str(data["token"]), None
        except httpx.TransportError as exc:  # includes timeouts, resets, incomplete chunked reads
            raise SellerInterrupted(f"{type(exc).__name__}: {exc or 'connection lost'}") from None
        except (ValueError, KeyError, TypeError) as exc:
            raise SellerRejected(f"malformed seller stream: {exc}") from None

    async def _replacement(self, needed: int) -> bool:
        """Reserve `needed` tokens elsewhere after a seller failure."""
        if self.failovers >= settings.max_failovers:
            return False
        plan = await reserve_capacity(
            buyer_id=self.buyer_id, instrument=self.instrument, tokens=needed, exclude=self.failed_trades
        )
        if plan is None and self.more_capacity is not None:
            plan = await self.more_capacity(needed, self.failed_trades)
        if not plan:
            return False
        start = len(self.segments)
        self.segments.extend(plan)
        self.queue[:0] = range(start, start + len(plan))
        await self.ckpt.save_segments(self.segments)
        return True

    # -- main loop -----------------------------------------------------------------
    async def _run(self):
        yield (
            "meta",
            {
                "job_id": self.job_id,
                "trade_id": self.seg.trade_id,
                "seller_id": self.seg.seller_id,
                "price_usd_per_mtok": npt_to_usd_per_mtok(self.seg.price_npt),
                "reserved_tokens": sum(s.reserved for s in self.segments),
                "segments_planned": len(self.segments),
            },
        )
        attempts_here = 0
        while self.finish_reason is None:
            seg = self.seg
            end = min(self.max_tokens, self.delivered + seg.reserved - seg.delivered)
            self.attempts += 1
            attempts_here += 1
            finish = None
            try:
                async with aclosing(self._call_seller(seg, end)) as stream:
                    async for index, token, done in stream:
                        if done is not None:
                            finish = done
                            break
                        if index < self.delivered:
                            continue  # replayed token after a resume; already delivered
                        if index > self.delivered or index >= end:
                            raise SellerRejected(f"token index {index} outside expected [{self.delivered}, {end})")
                        self.parts.append(token)
                        self.delivered += 1
                        seg.delivered += 1
                        await self.ckpt.add(index, token, self.current)
                        yield "token", {"index": index, "token": token}
                        if self.delivered >= end:
                            finish = "length"
                            break
                if finish is None:
                    raise SellerInterrupted("stream ended without completion marker (instance preempted)")
            except (SellerInterrupted, SellerRejected) as exc:
                await self.ckpt.flush()
                log.warning(
                    "job=%s seller=%s trade=%s interrupted at token %d (attempt %d): %s",
                    self.job_id,
                    seg.seller_id,
                    seg.trade_id,
                    self.delivered,
                    attempts_here,
                    exc,
                )
                if isinstance(exc, SellerInterrupted) and attempts_here < settings.max_attempts_per_allocation:
                    await asyncio.sleep(settings.retry_backoff_base_s * 2 ** (attempts_here - 1))
                    yield (
                        "resume",
                        {
                            "reason": str(exc),
                            "resume_from": self.delivered,
                            "trade_id": seg.trade_id,
                            "seller_id": seg.seller_id,
                            "failover": False,
                        },
                    )
                    continue
                # Give up on this allocation; its unused reservation is released at settlement.
                self.failed_trades.add(seg.trade_id)
                if not await self._replacement(seg.reserved - seg.delivered):
                    self.error = f"{exc}; no replacement capacity for {self.max_tokens - self.delivered} tokens"
                    return
                self.failovers += 1
                self.current = self.queue.pop(0)
                attempts_here = 0
                yield (
                    "resume",
                    {
                        "reason": str(exc),
                        "resume_from": self.delivered,
                        "trade_id": self.seg.trade_id,
                        "seller_id": self.seg.seller_id,
                        "price_usd_per_mtok": npt_to_usd_per_mtok(self.seg.price_npt),
                        "failover": True,
                    },
                )
                continue

            if finish == "length" and self.delivered < self.max_tokens and seg.delivered >= seg.reserved:
                # Segment capacity used up: roll over to the next planned allocation.
                if not self.queue:
                    self.error = "reserved capacity exhausted"
                    return
                self.current = self.queue.pop(0)
                attempts_here = 0
                continue
            self.finish_reason = finish

    async def _finalize(self) -> int | None:
        async with self._finalize_lock:
            if self._finalized:
                return self.cost
            self._finalized = True
            try:
                await self.ckpt.flush()
            except Exception:
                log.exception("job=%s checkpoint flush failed during settlement", self.job_id)
            if self.error is None and self.finish_reason is None:
                self.error = "client disconnected"
            self.cost = await settle_job(
                job_id=self.job_id,
                buyer_id=self.buyer_id,
                segments=self.segments,
                status=JOB_COMPLETED if self.finish_reason else JOB_FAILED,
                finish_reason=self.finish_reason,
                attempts=self.attempts,
                failovers=self.failovers,
                error=self.error,
            )
            try:
                await self.ckpt.finish()
            except Exception:
                log.exception("job=%s checkpoint finish failed", self.job_id)
            log.info(
                "SETTLED job=%s tokens=%d cost=%s segments=%s status=%s",
                self.job_id,
                self.delivered,
                fmt_usd(self.cost or 0),
                [(s.trade_id, s.delivered) for s in self.segments if s.delivered],
                JOB_COMPLETED if self.finish_reason else JOB_FAILED,
            )
            return self.cost

    def summary(self) -> dict:
        return {
            "job_id": self.job_id,
            "tokens": self.delivered,
            "cost_usd": fmt_usd(self.cost or 0),
            "cost_nanos": self.cost or 0,
            "attempts": self.attempts,
            "failovers": self.failovers,
            "segments": [
                {"trade_id": s.trade_id, "seller_id": s.seller_id, "tokens": s.delivered} for s in self.segments if s.delivered
            ],
        }

    async def events(self):
        settled = False
        try:
            async with aclosing(self._run()) as events:
                async for event in events:
                    yield event
            await self._finalize()
            settled = True
            if self.finish_reason:
                yield "done", self.summary() | {"finish_reason": self.finish_reason}
            else:
                yield "error", self.summary() | {"error": self.error}
        finally:
            if not settled:
                # Buyer disconnected or an unexpected error: settle in a task that
                # survives this generator's cancellation.
                task = asyncio.get_running_loop().create_task(self._finalize())
                _background.add(task)
                task.add_done_callback(_background.discard)
                await asyncio.shield(task)


async def start_run(
    request: Request,
    buyer_id: str,
    *,
    instrument: str,
    prompt: str,
    max_tokens: int,
    messages: list[dict] | None = None,
    trade_id: str | None = None,
    acquire: Callable[[int], Awaitable[list[Segment] | None]] | None = None,
    more_capacity: CapacitySource | None = None,
) -> InferenceRun:
    """Create the job, pass the escrow gate, and return a ready InferenceRun.
    Raises HTTP 402 if escrowed capacity cannot cover max_tokens."""
    job_id = new_id("job")
    async with SessionLocal() as session, session.begin():
        session.add(
            InferenceJob(
                id=job_id,
                buyer_id=buyer_id,
                instrument=instrument,
                tokens_requested=max_tokens,
                status=JOB_STREAMING,
                tokens_delivered=0,
                cost_nanos=0,
                attempts=0,
                failovers=0,
            )
        )

    async def fail(reason: str) -> None:
        await settle_job(
            job_id=job_id,
            buyer_id=buyer_id,
            segments=[],
            status=JOB_REJECTED,
            finish_reason=None,
            attempts=0,
            failovers=0,
            error=reason,
        )

    try:
        plan = await reserve_capacity(buyer_id=buyer_id, instrument=instrument, tokens=max_tokens, trade_id=trade_id)
        if plan is None and acquire is not None:
            plan = await acquire(max_tokens)
    except BaseException:
        await asyncio.shield(fail("capacity acquisition failed"))
        raise
    if plan is None:
        await fail("no escrowed capacity")
        raise HTTPException(
            402,
            detail={
                "error": "no_escrowed_allocation",
                "error_description": f"escrowed allocations of {instrument} cannot cover {max_tokens} tokens; "
                "buy inference units via POST /v1/orders first",
            },
        )

    run = InferenceRun(
        request.app.state.http,
        request.app.state.redis,
        buyer_id,
        job_id,
        instrument=instrument,
        prompt=prompt,
        max_tokens=max_tokens,
        plan=plan,
        messages=messages,
        more_capacity=more_capacity,
    )
    try:
        await run.ckpt.save_segments(run.segments)
    except Exception:
        await run._finalize()
        raise
    log.info(
        "ROUTE job=%s buyer=%s -> %s",
        job_id,
        buyer_id,
        [(s.seller_id, s.trade_id, s.reserved) for s in plan],
    )
    return run


# ---------------------------------------------------------------------- routes


@router.post("/v1/inference", response_class=StreamingResponse)
async def inference(req: InferenceRequest, request: Request, ctx: AuthContext = Depends(require_scopes(SCOPE_BUY))):
    run = await start_run(
        request,
        ctx.agent_id,
        instrument=req.instrument,
        prompt=req.prompt,
        max_tokens=req.max_tokens,
        trade_id=req.trade_id,
    )

    async def body():
        async with aclosing(run.events()) as events:
            async for kind, data in events:
                yield sse(data, None if kind == "token" else kind)

    return StreamingResponse(
        body(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "X-Aether-Job-Id": run.job_id},
    )


@router.get("/v1/jobs/{job_id}")
async def get_job(job_id: str, request: Request, ctx: AuthContext = Depends(require_scopes(SCOPE_BUY))):
    """Job status plus the checkpointed output, so a buyer whose own connection
    dropped can recover what was already generated (and paid for)."""
    async with SessionLocal() as session:
        job = await session.get(InferenceJob, job_id)
    if job is None or job.buyer_id != ctx.agent_id:
        raise HTTPException(404, detail="job not found")
    tokens = await read_checkpoint(request.app.state.redis, job_id)
    return {
        "job_id": job.id,
        "status": job.status,
        "finish_reason": job.finish_reason,
        "tokens_requested": job.tokens_requested,
        "tokens_delivered": job.tokens_delivered if job.status != JOB_STREAMING else len(tokens),
        "cost_usd": fmt_usd(job.cost_nanos),
        "attempts": job.attempts,
        "failovers": job.failovers,
        "segments": json.loads(job.segments) if job.segments else None,
        "error": job.error,
        "checkpointed_text": "".join(t[1] for t in tokens),
    }


# --------------------------------------------------------------- orphan recovery


async def recover_orphaned_jobs(redis: Redis, min_age_s: int = 60) -> int:
    """Settle jobs whose proxy process died mid-stream (no heartbeat), paying
    sellers only for checkpointed tokens and releasing the reservations."""
    cutoff = utcnow() - timedelta(seconds=min_age_s)
    async with SessionLocal() as session:
        jobs = (
            (
                await session.execute(
                    select(InferenceJob).where(InferenceJob.status == JOB_STREAMING, InferenceJob.created_at < cutoff).limit(50)
                )
            )
            .scalars()
            .all()
        )
    recovered = 0
    for job in jobs:
        prefix = f"{settings.redis_prefix}:job:{job.id}"
        if await redis.exists(f"{prefix}:hb"):
            continue
        raw = await redis.get(f"{prefix}:segments")
        if raw is None:
            log.error("orphaned job %s has no segment record; reservations need manual repair", job.id)
            continue
        segments = [Segment(**s, delivered=0) for s in json.loads(raw)]
        for _, _, segment in await read_checkpoint(redis, job.id):
            if 0 <= segment < len(segments):
                segments[segment].delivered += 1
        await settle_job(
            job_id=job.id,
            buyer_id=job.buyer_id,
            segments=segments,
            status=JOB_FAILED,
            finish_reason=None,
            attempts=0,
            failovers=0,
            error="proxy stopped mid-stream; settled from checkpoint",
        )
        recovered += 1
    return recovered


async def drain_background(timeout: float = 10.0) -> None:
    if _background:
        await asyncio.wait(list(_background), timeout=timeout)
