"""Production seller agent: puts a GPU box on the Aether market.

Run it next to any OpenAI-compatible inference server you operate (vLLM,
SGLang, TGI, llama.cpp server, Ollama):

    AETHER_URL=https://clearinghouse.example.com \\
    SELLER_API_KEY=agt_cli_xxx.aes_yyy \\
    SELLER_PUBLIC_URL=https://gpu1.example.com/v1/generate \\
    UPSTREAM_BASE_URL=http://127.0.0.1:8001/v1 \\
    UPSTREAM_MODEL=meta-llama/Llama-3.1-70B-Instruct \\
    INSTRUMENT=llama-3.1-70b-instruct \\
    PRICE_USD_PER_MTOK=0.35 CAPACITY_TOKENS=2000000 \\
    python -m app.seller_gateway

It registers its endpoint, keeps a GTC ask of CAPACITY_TOKENS resting on the
book (re-listing as capacity is sold or expires), verifies every delivery JWT
offline against the clearinghouse JWKS (audience, single use, body hash), and
relays the upstream token stream in the clearinghouse's SSE protocol,
including resume-from-checkpoint (prefix continuation).

Only serve models you have the right to serve. Fronting a proprietary API
(OpenAI, Anthropic, Google...) with your own key is resale of that access and
is refused unless ALLOW_PROPRIETARY_UPSTREAM=true (for providers that have
explicitly licensed resale to you).
"""

import asyncio
import hashlib
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from urllib.parse import urlparse

import httpx
import jwt
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

log = logging.getLogger("seller-gateway")

PROPRIETARY_HOSTS = ("api.openai.com", "api.anthropic.com", "generativelanguage.googleapis.com", "aiplatform.googleapis.com")


class Config:
    def __init__(self):
        env = os.environ.get
        self.clearinghouse = env("AETHER_URL", "http://localhost:8000").rstrip("/")
        self.api_key = env("SELLER_API_KEY", "")
        self.public_url = env("SELLER_PUBLIC_URL", "")
        self.port = int(env("PORT", "9100"))
        self.upstream = env("UPSTREAM_BASE_URL", "http://127.0.0.1:8001/v1").rstrip("/")
        self.upstream_key = env("UPSTREAM_API_KEY", "")
        self.upstream_model = env("UPSTREAM_MODEL", "")
        self.instrument = env("INSTRUMENT", "")
        self.price = env("PRICE_USD_PER_MTOK", "0.35")
        self.capacity = int(env("CAPACITY_TOKENS", "1000000"))
        self.ask_ttl = int(env("ASK_TTL_S", "3600"))
        self.relist_interval = float(env("RELIST_INTERVAL_S", "30"))
        self.chat_continuation = env("CHAT_CONTINUATION", "vllm")  # vllm | none
        self.allow_proprietary = env("ALLOW_PROPRIETARY_UPSTREAM", "false").lower() == "true"


class State:
    def __init__(self):
        self.agent_id: str | None = None
        self.issuer: str | None = None
        self.keys: dict[str, object] = {}
        self.seen_jti: dict[str, float] = {}
        self.order_id: str | None = None


cfg = Config()
state = State()


def check_config(c: Config) -> None:
    missing = [
        n
        for n, v in [
            ("SELLER_API_KEY", c.api_key),
            ("SELLER_PUBLIC_URL", c.public_url),
            ("UPSTREAM_MODEL", c.upstream_model),
            ("INSTRUMENT", c.instrument),
        ]
        if not v
    ]
    if missing:
        raise SystemExit(f"missing required settings: {', '.join(missing)}")
    host = urlparse(c.upstream).hostname or ""
    if not c.allow_proprietary and any(host == h or host.endswith("." + h) for h in PROPRIETARY_HOSTS):
        raise SystemExit(
            f"refusing to resell {host}: serve a model you operate, or set ALLOW_PROPRIETARY_UPSTREAM=true "
            "only if that provider licenses resale to you"
        )


def _auth() -> dict:
    return {"Authorization": f"Bearer {cfg.api_key}"}


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


def upstream_request(body: dict) -> tuple[str, dict]:
    """Translate a clearinghouse job segment into an upstream streaming call.
    Resumes continue from `prefix` so the output is seamless."""
    remaining = body["max_tokens"] - body["resume_from"]
    prefix = body.get("prefix", "")
    messages = body.get("messages")
    if messages:
        payload: dict = {"model": cfg.upstream_model, "messages": list(messages), "max_tokens": remaining, "stream": True}
        if prefix:
            payload["messages"].append({"role": "assistant", "content": prefix})
            if cfg.chat_continuation == "vllm":  # vLLM / SGLang: continue the partial assistant turn
                payload["add_generation_prompt"] = False
                payload["continue_final_message"] = True
        return "/chat/completions", payload
    return "/completions", {
        "model": cfg.upstream_model,
        "prompt": body["prompt"] + prefix,
        "max_tokens": remaining,
        "stream": True,
    }


async def relay(http: httpx.AsyncClient, body: dict):
    path, payload = upstream_request(body)
    index = body["resume_from"]
    end = body["max_tokens"]
    headers = {"Authorization": f"Bearer {cfg.upstream_key}"} if cfg.upstream_key else {}
    finish = "stop"
    async with http.stream("POST", cfg.upstream + path, json=payload, headers=headers) as resp:
        if resp.status_code != 200:
            detail = (await resp.aread())[:200].decode(errors="replace")
            log.error("upstream HTTP %s: %s", resp.status_code, detail)
            return  # no completion marker -> clearinghouse retries/fails over
        async for line in resp.aiter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            choice = (json.loads(data).get("choices") or [{}])[0]
            text = choice.get("text") if "text" in choice else (choice.get("delta") or {}).get("content")
            if text:
                # One upstream chunk is metered as one token (vLLM/SGLang stream per token);
                # a server that batches tokens per chunk under-bills, never over-bills.
                yield f"data: {json.dumps({'index': index, 'token': text})}\n\n"
                index += 1
                if index >= end:
                    finish = "length"
                    break
            if choice.get("finish_reason"):
                finish = "length" if choice["finish_reason"] == "length" else "stop"
    yield f"data: {json.dumps({'done': True, 'finish_reason': finish})}\n\n"


async def ensure_listed(client: httpx.AsyncClient) -> None:
    """Keep one ask resting with fresh capacity; replace it when mostly sold or gone."""
    if state.order_id:
        r = await client.get(f"/v1/orders/{state.order_id}", headers=_auth())
        if r.status_code == 200:
            order = r.json()
            if order["status"] == "open" and order["open_tokens"] >= cfg.capacity // 4:
                return
            if order["status"] == "open":
                await client.delete(f"/v1/orders/{state.order_id}", headers=_auth())
    r = await client.post(
        "/v1/orders",
        headers=_auth(),
        json={
            "instrument": cfg.instrument,
            "side": "ask",
            "order_type": "limit",
            "time_in_force": "gtc",
            "price_usd_per_mtok": cfg.price,
            "quantity_tokens": cfg.capacity,
            "ttl_seconds": cfg.ask_ttl,
        },
    )
    r.raise_for_status()
    state.order_id = r.json()["order"]["order_id"]
    log.info("ASK listed %s: %d tokens @ $%s/1M (%s)", cfg.instrument, cfg.capacity, cfg.price, state.order_id)


async def relist_loop(client: httpx.AsyncClient) -> None:
    while True:
        await asyncio.sleep(cfg.relist_interval)
        try:
            await ensure_listed(client)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("re-listing failed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    check_config(cfg)
    client = httpx.AsyncClient(base_url=cfg.clearinghouse, timeout=15)
    app.state.upstream = httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=120.0))
    me = (await client.get("/v1/agents/me", headers=_auth())).raise_for_status().json()
    state.agent_id = me["agent_id"]
    state.issuer = (await client.get("/.well-known/oauth-authorization-server")).json()["issuer"]
    (await client.put("/v1/agents/me/endpoint", headers=_auth(), json={"endpoint_url": cfg.public_url})).raise_for_status()
    await refresh_jwks(client)
    await ensure_listed(client)
    task = asyncio.create_task(relist_loop(client))
    log.info("seller %s serving %s via %s", state.agent_id, cfg.instrument, cfg.upstream)
    try:
        yield
    finally:
        task.cancel()
        if state.order_id:
            try:
                await client.delete(f"/v1/orders/{state.order_id}", headers=_auth())  # stop selling on shutdown
            except httpx.HTTPError:
                pass
        await client.aclose()
        await app.state.upstream.aclose()


app = FastAPI(title="Aether seller gateway", lifespan=lifespan)


@app.get("/healthz")
async def healthz():
    return {"status": "ok", "agent_id": state.agent_id, "order_id": state.order_id}


@app.post("/v1/generate")
async def generate(request: Request):
    auth = request.headers.get("authorization", "")
    if not auth.lower().startswith("bearer "):
        raise HTTPException(401, detail="missing delivery token")
    raw = await request.body()
    claims = await verify_delivery_token(auth[7:], raw)
    body = json.loads(raw)
    if (body["job_id"], body["max_tokens"], body["resume_from"]) != (claims["sub"], claims["max_tokens"], claims["resume_from"]):
        raise HTTPException(401, detail="request does not match delivery token claims")
    if body["instrument"] != cfg.instrument:
        raise HTTPException(400, detail=f"this seller serves {cfg.instrument}")
    return StreamingResponse(relay(request.app.state.upstream, body), media_type="text/event-stream")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    uvicorn.run(app, host="0.0.0.0", port=cfg.port, log_level="warning")


if __name__ == "__main__":
    main()
