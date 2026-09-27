"""OpenAI-compatible surface: the zero-integration path for agents.

Any agent built on an OpenAI-style client (OpenAI SDK, LangChain, LlamaIndex,
CrewAI, AutoGen, Vercel AI SDK, ...) joins the market by changing two settings:

    base_url = "https://<clearinghouse>/v1"
    api_key  = "<client_id>.<client_secret>"      (or a short-lived JWT)

`model` names the instrument. Each request is served from the agent's escrowed
allocations; if they cannot cover `max_tokens`, the deficit is market-bought
on the spot (IOC) under a price cap — the agent's `max_price_usd_per_mtok`
(sent via `extra_body`) or the platform default. Metering, escrow,
checkpoint-resume/failover and settlement are identical to /v1/inference.

Pricing today covers completion tokens only; prompt tokens are not billed and
are reported as 0 in `usage`.
"""

import time
from contextlib import aclosing
from decimal import Decimal
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from .auth import SCOPE_BUY, AuthContext, require_scopes
from .config import settings
from .exchange import ExchangeError, Segment, available_capacity, buy_capacity, reserve_capacity
from .proxy_router import INSTRUMENT_PATTERN, sse, start_run
from .units import NPT_PER_USD_PER_MTOK, fmt_usd, npt_to_usd_per_mtok, usd_per_mtok_to_npt

router = APIRouter(prefix="/v1", tags=["openai-compatible"])


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: str
    content: str | list[dict[str, Any]] | None = None


class StreamOptions(BaseModel):
    include_usage: bool = False


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")  # tolerate unsupported OpenAI params

    model: str = Field(pattern=INSTRUMENT_PATTERN)
    messages: list[ChatMessage] = Field(min_length=1)
    max_tokens: int | None = Field(None, gt=0, le=settings.max_tokens_per_request)
    max_completion_tokens: int | None = Field(None, gt=0, le=settings.max_tokens_per_request)
    stream: bool = False
    stream_options: StreamOptions | None = None
    # Aether extension (OpenAI SDK: extra_body={"max_price_usd_per_mtok": "0.50"})
    max_price_usd_per_mtok: Decimal | None = Field(None, gt=0)
    auto_buy: bool = True


def _text(content: str | list[dict[str, Any]] | None) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content if part.get("type") == "text")


def render_prompt(messages: list[ChatMessage]) -> str:
    """Plain-text fallback for sellers that do not apply a chat template;
    template-aware sellers use the structured `messages` field instead."""
    lines = [f"{m.role}: {_text(m.content)}" for m in messages]
    return "\n".join(lines) + "\nassistant:"


def _chunk(cid: str, created: int, model: str, delta: dict, finish: str | None = None) -> dict:
    return {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


def _usage(tokens: int) -> dict:
    return {"prompt_tokens": 0, "completion_tokens": tokens, "total_tokens": tokens}


def _openai_error(status: int, message: str, code: str) -> HTTPException:
    return HTTPException(status, detail={"error": {"message": message, "type": code, "code": code}})


@router.post("/chat/completions")
async def chat_completions(req: ChatCompletionRequest, request: Request, ctx: AuthContext = Depends(require_scopes(SCOPE_BUY))):
    max_tokens = req.max_completion_tokens or req.max_tokens or settings.default_completion_tokens
    try:
        cap_npt = usd_per_mtok_to_npt(req.max_price_usd_per_mtok or settings.default_max_price_usd_per_mtok)
    except ValueError as exc:
        raise _openai_error(422, str(exc), "invalid_request_error") from None
    engine, redis = request.app.state.engine, request.app.state.redis

    async def buy_and_reserve(tokens: int, exclude: set[str] | frozenset = frozenset()) -> list[Segment] | None:
        if not req.auto_buy:
            return None
        have = await available_capacity(buyer_id=ctx.agent_id, instrument=req.model, exclude=exclude)
        try:
            await buy_capacity(
                engine, redis, buyer_id=ctx.agent_id, instrument=req.model, tokens=max(tokens - have, 1), max_price_npt=cap_npt
            )
        except ExchangeError as exc:
            if exc.status_code == 402:
                raise _openai_error(
                    402, f"insufficient balance to escrow {tokens} tokens at the price cap", "insufficient_quota"
                ) from None
            raise _openai_error(exc.status_code, exc.message, "invalid_request_error") from None
        return await reserve_capacity(buyer_id=ctx.agent_id, instrument=req.model, tokens=tokens, exclude=exclude)

    messages = [m.model_dump(exclude_none=True) for m in req.messages]
    try:
        run = await start_run(
            request,
            ctx.agent_id,
            instrument=req.model,
            prompt=render_prompt(req.messages),
            max_tokens=max_tokens,
            messages=messages,
            acquire=buy_and_reserve,
            more_capacity=buy_and_reserve,
        )
    except HTTPException as exc:
        detail = exc.detail if isinstance(exc.detail, dict) else {}
        if detail.get("error") != "no_escrowed_allocation":
            raise
        if req.auto_buy:
            raise _openai_error(
                503,
                f"not enough {req.model} liquidity at or below ${npt_to_usd_per_mtok(cap_npt)}/1M tokens "
                f"for {max_tokens} tokens; raise max_price_usd_per_mtok or retry",
                "insufficient_liquidity",
            ) from None
        raise _openai_error(402, detail["error_description"], "insufficient_quota") from None

    cid = f"chatcmpl-{run.job_id}"
    created = int(time.time())
    headers = {"X-Aether-Job-Id": run.job_id}

    if req.stream:
        include_usage = bool(req.stream_options and req.stream_options.include_usage)

        async def body():
            yield sse(_chunk(cid, created, req.model, {"role": "assistant", "content": ""}))
            async with aclosing(run.events()) as events:
                async for kind, data in events:
                    if kind == "token":
                        yield sse(_chunk(cid, created, req.model, {"content": data["token"]}))
                    elif kind == "done":
                        yield sse(_chunk(cid, created, req.model, {}, data["finish_reason"]))
                        if include_usage:
                            final = _chunk(cid, created, req.model, {})
                            final["choices"] = []
                            final["usage"] = _usage(data["tokens"])
                            final["aether"] = {"cost_usd": data["cost_usd"], "segments": data["segments"]}
                            yield sse(final)
                    elif kind == "error":
                        yield sse({"error": {"message": data["error"], "type": "upstream_error", "code": "seller_failure"}})
            yield b"data: [DONE]\n\n"

        return StreamingResponse(
            body(), media_type="text/event-stream", headers=headers | {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
        )

    text: list[str] = []
    result: dict = {}
    async with aclosing(run.events()) as events:
        async for kind, data in events:
            if kind == "token":
                text.append(data["token"])
            elif kind in ("done", "error"):
                result = data | {"kind": kind}
    if result.get("kind") != "done":
        raise _openai_error(
            502, f"inference failed after {result.get('tokens', 0)} tokens: {result.get('error')}", "upstream_error"
        )
    return {
        "id": cid,
        "object": "chat.completion",
        "created": created,
        "model": req.model,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "".join(text)}, "finish_reason": result["finish_reason"]}
        ],
        "usage": _usage(result["tokens"]),
        "aether": {"job_id": run.job_id, "cost_usd": result["cost_usd"], "segments": result["segments"]},
    }


@router.get("/models")
async def list_models(request: Request):
    """OpenAI-style model list: every instrument that has had sell-side
    liquidity, with its current best ask and depth."""
    redis, engine = request.app.state.redis, request.app.state.engine
    data = []
    for instrument in sorted(await redis.smembers(f"{settings.redis_prefix}:instruments")):
        depth = await engine.depth(instrument, levels=50)
        asks = depth["asks"]
        data.append(
            {
                "id": instrument,
                "object": "model",
                "created": 0,
                "owned_by": "aether-market",
                "best_ask_usd_per_mtok": npt_to_usd_per_mtok(asks[0][0]) if asks else None,
                "ask_depth_tokens": sum(q for _, q, _ in asks),
            }
        )
    return {"object": "list", "data": data}


class QuoteOut(BaseModel):
    model: str
    tokens: int
    fillable_tokens: int
    estimated_cost_usd: str
    average_price_usd_per_mtok: str | None
    side: Literal["buy"] = "buy"


@router.get("/quote/{model}", response_model=QuoteOut)
async def quote(model: str, request: Request, tokens: int = 1000):
    """Walk the ask book to price `tokens` before committing (no reservation)."""
    depth = await request.app.state.engine.depth(model, levels=100)
    need, cost, filled = tokens, 0, 0
    for price, qty, _ in depth["asks"]:
        take = min(qty, need)
        cost += take * price
        filled += take
        need -= take
        if need == 0:
            break
    return QuoteOut(
        model=model,
        tokens=tokens,
        fillable_tokens=filled,
        estimated_cost_usd=fmt_usd(cost),
        average_price_usd_per_mtok=str((Decimal(cost) / filled / NPT_PER_USD_PER_MTOK).quantize(Decimal("0.000001")).normalize())
        if filled
        else None,
    )
