"""FastAPI entry point for the Aether clearinghouse."""

import asyncio
import logging
from contextlib import asynccontextmanager, suppress

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from redis.asyncio import ConnectionPool, Redis
from sqlalchemy import text

from . import api, auth, crypto_payments, feedback, openai_compat, payments, proxy_router, services, site
from .config import settings
from .db import engine as db_engine
from .db import init_models
from .exchange import SettlementWorker, reconcile_book, run_sweeps
from .matching_engine import MatchingEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
log = logging.getLogger("aether")


async def maintenance_loop(app: FastAPI) -> None:
    while True:
        try:
            await run_sweeps(app.state.engine, app.state.redis)
            await proxy_router.recover_orphaned_jobs(app.state.redis)
            await crypto_payments.run_watcher_pass(app)
            await services.refund_stale_calls()
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("maintenance pass failed")
        await asyncio.sleep(settings.sweep_interval_s)


@asynccontextmanager
async def lifespan(app: FastAPI):
    auth.init_signing_key()
    await init_models()

    pool = ConnectionPool.from_url(settings.redis_url, max_connections=settings.redis_max_connections, decode_responses=True)
    redis = Redis(connection_pool=pool)
    await redis.ping()
    app.state.redis = redis
    app.state.engine = MatchingEngine(
        redis, settings.redis_prefix, max_fills=settings.max_fills_per_match, stream_maxlen=settings.match_stream_maxlen
    )
    app.state.http = httpx.AsyncClient(
        limits=httpx.Limits(
            max_connections=settings.http_max_connections, max_keepalive_connections=settings.http_max_connections // 4
        ),
        follow_redirects=False,  # never let a seller bounce the delivery token elsewhere
    )

    app.state.stripe = payments.StripeClient.create()
    app.state.crypto = crypto_payments.CryptoRails.create()

    # Crash recovery: apply unapplied match events, then make the Redis book
    # agree with PostgreSQL before accepting orders.
    worker = SettlementWorker(redis, app.state.engine)
    await worker.drain()
    await reconcile_book(app.state.engine)

    tasks = [
        asyncio.create_task(worker.run(), name="settlement-worker"),
        asyncio.create_task(maintenance_loop(app), name="maintenance"),
    ]
    log.info(
        "Aether clearinghouse ready (sandbox=%s, fee=%sbps, stripe=%s, crypto=%s)",
        settings.sandbox_mode,
        settings.fee_bps,
        "on" if app.state.stripe else "off",
        ",".join(f"{r.network.token_symbol}@{r.network.key}" for r in app.state.crypto.rails.values())
        if app.state.crypto
        else "off",
    )
    try:
        yield
    finally:
        for t in tasks:
            t.cancel()
        for t in tasks:
            with suppress(asyncio.CancelledError):
                await t
        await proxy_router.drain_background()
        await app.state.http.aclose()
        if app.state.stripe is not None:
            await app.state.stripe.aclose()
        if app.state.crypto is not None:
            await app.state.crypto.aclose()
        await redis.aclose()
        await pool.disconnect()
        await db_engine.dispose()


app = FastAPI(
    title="Project Aether — M2M Inference Clearinghouse",
    version="0.1.0",
    description="Agents trade abstracted inference units through an escrowed spot market; credentials never change hands.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
    expose_headers=["X-Aether-Job-Id"],
)
app.include_router(auth.router)
app.include_router(api.router)
app.include_router(proxy_router.router)
app.include_router(openai_compat.router)
app.include_router(payments.router)
app.include_router(crypto_payments.router)
app.include_router(services.router)
app.include_router(feedback.router)
app.include_router(site.router)


@app.get("/healthz", tags=["ops"])
async def healthz(request: Request):
    try:
        await request.app.state.redis.ping()
        async with db_engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception as exc:  # pragma: no cover - surfaced to the orchestrator
        return JSONResponse({"status": "degraded", "error": str(exc)}, status_code=503)
    return {"status": "ok"}
