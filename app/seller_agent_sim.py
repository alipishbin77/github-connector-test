"""Mock Seller Agent.

Boots, registers itself with the clearinghouse, obtains a `sell_compute` JWT
via the client-credentials grant, registers its stateless inference webhook,
and lists spot capacity as a GTC ask. It then serves a deterministic mock LLM
over SSE.

Every request must carry a clearinghouse-signed delivery JWT, verified
offline against the JWKS: audience == this seller, single use (jti), and bound
to the exact request body by SHA-256. The seller never sees a buyer credential
and never exposes one of its own.

Chaos: PREEMPT_AFTER_TOKENS / PREEMPT_ATTEMPTS simulate a spot instance being
reclaimed mid-stream (the stream ends without its completion marker), which
exercises the proxy's checkpoint-resume and failover paths.

Run:  python -m app.seller_agent_sim
"""

import asyncio
import hashlib
import json
import logging
import os
import time
from contextlib import asynccontextmanager

import httpx
import jwt
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

log = logging.getLogger("seller")

CORPUS = [
    "Spot capacity is perishable, so an idle accelerator is revenue that evaporates every second.",
    "The clearinghouse matched the bid against the cheapest resting ask and escrowed the difference.",
    "No credential ever crossed the wire; only a signed, single-use delivery token did.",
    "When the instance was reclaimed, the stream resumed from the last checkpoint on another seller.",
    "Every inference unit was settled in nano-dollars against the buyer's escrow, net of the clearing fee.",
    "Machines negotiate in prices and latencies, and the order book is their common language.",
    "Stateless workers make preemption cheap: the checkpoint is the only state that matters.",
    "Price-time priority means the cheapest and earliest capacity is consumed first.",
]


class Config:
    def __init__(self):
        env = os.environ.get
        self.clearinghouse = env("CLEARINGHOUSE_URL", "http://localhost:8000").rstrip("/")
        self.name = env("SELLER_NAME", "seller-sim")
        self.port = int(env("SELLER_PORT", "9001"))
        self.public_url = env("SELLER_PUBLIC_URL", f"http://localhost:{self.port}/v1/generate")
        self.instrument = env("INSTRUMENT", "llama-3.1-70b-instruct")
        self.price = env("PRICE_USD_PER_MTOK", "0.35")
        self.capacity = int(env("CAPACITY_TOKENS", "5000"))
        self.ask_ttl = int(env("ASK_TTL_S", "3600"))
        self.preempt_after = int(env("PREEMPT_AFTER_TOKENS", "0"))
        self.preempt_attempts = int(env("PREEMPT_ATTEMPTS", "1"))
        self.token_delay = float(env("TOKEN_DELAY_MS", "15")) / 1000


cfg = Config()


class SellerState:
    def __init__(self):
        self.agent_id: str | None = None
        self.issuer: str | None = None
        self.keys: dict[str, object] = {}
        self.seen_jti: dict[str, float] = {}  # replay protection, pruned at exp
        self.attempts: dict[str, int] = {}  # per-job attempt counter for chaos


state = SellerState()


def tokens_for(prompt: str, count: int) -> list[str]:
    """Deterministic 'model': the same prompt yields the same token sequence on
    any replica, so a resumed or failed-over job continues seamlessly. A real
    engine would instead continue generation from `prompt + prefix`."""
    start = int(hashlib.sha256(prompt.encode()).hexdigest(), 16) % len(CORPUS)
    words: list[str] = []
    i = start
    while len(words) < count:
        words.extend(CORPUS[i % len(CORPUS)].split())
        i += 1
    return [(" " if n else "") + w for n, w in enumerate(words[:count])]


async def refresh_jwks(client: httpx.AsyncClient) -> None:
    resp = await client.get("/.well-known/jwks.json")
    resp.raise_for_status()
    state.keys = {k["kid"]: jwt.PyJWK(k).key for k in resp.json()["keys"]}


async def verify_delivery_token(token: str, raw_body: bytes) -> dict:
    try:
        kid = jwt.get_unverified_header(token).get("kid")
        if kid not in state.keys:  # key rotation: refetch once
            async with httpx.AsyncClient(base_url=cfg.clearinghouse, timeout=5) as c:
                await refresh_jwks(c)
        claims = jwt.decode(
            token,
            state.keys[kid],
            algorithms=["RS256"],
            audience=f"aether-seller:{state.agent_id}",
            issuer=state.issuer,
            options={"require": ["exp", "iat", "jti", "aud", "iss", "sub"]},
        )
    except (jwt.InvalidTokenError, KeyError) as exc:
        raise HTTPException(401, detail=f"invalid delivery token: {exc}") from None
    if claims.get("typ") != "delivery":
        raise HTTPException(401, detail="not a delivery token")
    now = time.time()
    for jti, exp in list(state.seen_jti.items()):
        if exp < now:
            del state.seen_jti[jti]
    if claims["jti"] in state.seen_jti:
        raise HTTPException(401, detail="delivery token replayed")
    state.seen_jti[claims["jti"]] = claims["exp"]
    if hashlib.sha256(raw_body).hexdigest() != claims["body_sha256"]:
        raise HTTPException(401, detail="request body does not match delivery token")
    return claims


async def bootstrap() -> None:
    async with httpx.AsyncClient(base_url=cfg.clearinghouse, timeout=15) as c:
        for _ in range(60):
            try:
                if (await c.get("/healthz")).status_code == 200:
                    break
            except httpx.TransportError:
                pass
            await asyncio.sleep(1)
        else:
            raise RuntimeError("clearinghouse never became healthy")

        meta = (await c.get("/.well-known/oauth-authorization-server")).json()
        state.issuer = meta["issuer"]
        reg = (await c.post("/v1/agents", json={"name": cfg.name, "scopes": ["sell_compute"]})).raise_for_status().json()
        state.agent_id = reg["agent_id"]
        log.info("[%s] registered agent_id=%s client_id=%s", cfg.name, reg["agent_id"], reg["client_id"])

        tok = (
            (
                await c.post(
                    "/oauth/token",
                    data={"grant_type": "client_credentials", "scope": "sell_compute"},
                    auth=(reg["client_id"], reg["client_secret"]),
                )
            )
            .raise_for_status()
            .json()
        )
        headers = {"Authorization": f"Bearer {tok['access_token']}"}
        log.info("[%s] OAuth2 client_credentials -> JWT scope=%r expires_in=%ss", cfg.name, tok["scope"], tok["expires_in"])

        await c.put("/v1/agents/me/endpoint", json={"endpoint_url": cfg.public_url}, headers=headers)
        await refresh_jwks(c)
        log.info("[%s] endpoint registered: %s (JWKS keys cached: %d)", cfg.name, cfg.public_url, len(state.keys))

        order = (
            (
                await c.post(
                    "/v1/orders",
                    json={
                        "instrument": cfg.instrument,
                        "side": "ask",
                        "order_type": "limit",
                        "time_in_force": "gtc",
                        "price_usd_per_mtok": cfg.price,
                        "quantity_tokens": cfg.capacity,
                        "ttl_seconds": cfg.ask_ttl,
                    },
                    headers=headers,
                )
            )
            .raise_for_status()
            .json()["order"]
        )
        log.info(
            "[%s] ASK listed %s: %d tokens @ $%s/1M (order %s, status %s)",
            cfg.name,
            cfg.instrument,
            cfg.capacity,
            cfg.price,
            order["order_id"],
            order["status"],
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    await bootstrap()
    yield


app = FastAPI(title="Aether mock seller agent", lifespan=lifespan)


@app.get("/healthz")
async def healthz():
    return {"status": "ok", "agent_id": state.agent_id}


@app.post("/v1/generate")
async def generate(request: Request):
    auth = request.headers.get("authorization", "")
    if not auth.lower().startswith("bearer "):
        raise HTTPException(401, detail="missing delivery token")
    raw = await request.body()
    claims = await verify_delivery_token(auth[7:], raw)
    body = json.loads(raw)
    if (
        body["job_id"] != claims["sub"]
        or body["max_tokens"] != claims["max_tokens"]
        or body["resume_from"] != claims["resume_from"]
    ):
        raise HTTPException(401, detail="request does not match delivery token claims")
    if body["instrument"] != cfg.instrument:
        raise HTTPException(400, detail=f"this seller serves {cfg.instrument}")

    job_id, start, end = body["job_id"], body["resume_from"], body["max_tokens"]
    attempt = state.attempts.get(job_id, 0)
    state.attempts[job_id] = attempt + 1
    tokens = tokens_for(body["prompt"], end)
    preempt_at = start + cfg.preempt_after if cfg.preempt_after and attempt < cfg.preempt_attempts else None
    log.info(
        "[%s] job=%s attempt=%d serving tokens [%d, %d) trade=%s prefix_chars=%d",
        cfg.name,
        job_id,
        attempt + 1,
        start,
        end,
        claims["trade_id"],
        len(body.get("prefix", "")),
    )

    async def stream():
        for i in range(start, end):
            if preempt_at is not None and i >= preempt_at:
                log.warning("[%s] job=%s SIMULATED SPOT PREEMPTION after token %d", cfg.name, job_id, i - 1)
                return  # connection closes without the completion marker
            await asyncio.sleep(cfg.token_delay)
            yield f"data: {json.dumps({'index': i, 'token': tokens[i]})}\n\n"
        yield f"data: {json.dumps({'done': True, 'finish_reason': 'length'})}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    uvicorn.run(app, host="0.0.0.0", port=cfg.port, log_level="warning")


if __name__ == "__main__":
    main()
