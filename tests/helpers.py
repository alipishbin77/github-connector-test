import hashlib
import json

import httpx
import jwt

from app import auth


async def new_agent(client: httpx.AsyncClient, name: str, scopes: list[str], fund_usd: str | None = None) -> dict:
    reg = (await client.post("/v1/agents", json={"name": name, "scopes": scopes})).raise_for_status().json()
    tok = (
        (
            await client.post(
                "/oauth/token",
                data={"grant_type": "client_credentials"},
                auth=(reg["client_id"], reg["client_secret"]),
            )
        )
        .raise_for_status()
        .json()
    )
    agent = reg | {"headers": {"Authorization": f"Bearer {tok['access_token']}"}, "token": tok["access_token"]}
    if fund_usd:
        (await client.post("/v1/sandbox/faucet", headers=agent["headers"], json={"amount_usd": fund_usd})).raise_for_status()
    return agent


async def new_seller(client, name: str, endpoint: str) -> dict:
    seller = await new_agent(client, name, ["sell_compute"])
    (await client.put("/v1/agents/me/endpoint", headers=seller["headers"], json={"endpoint_url": endpoint})).raise_for_status()
    return seller


async def ask(client, seller, instrument, price, qty, **extra):
    r = await client.post(
        "/v1/orders",
        headers=seller["headers"],
        json={"instrument": instrument, "side": "ask", "price_usd_per_mtok": price, "quantity_tokens": qty} | extra,
    )
    r.raise_for_status()
    return r.json()


async def bid(client, buyer, instrument, price, qty, **extra):
    r = await client.post(
        "/v1/orders",
        headers=buyer["headers"],
        json={"instrument": instrument, "side": "bid", "price_usd_per_mtok": price, "quantity_tokens": qty} | extra,
    )
    return r


async def me(client, agent) -> dict:
    return (await client.get("/v1/agents/me", headers=agent["headers"])).raise_for_status().json()


async def audit_ok(client) -> dict:
    result = (await client.get("/v1/audit")).json()
    assert result["ok"], result
    return result


def parse_sse(body: str) -> list[tuple[str | None, dict]]:
    events, name = [], None
    for line in body.splitlines():
        if line.startswith("event:"):
            name = line[6:].strip()
        elif line.startswith("data:"):
            events.append((name, json.loads(line[5:])))
            name = None
    return events


class MockSellers:
    """httpx transport standing in for seller endpoints. Verifies every
    delivery token exactly like a real seller must."""

    def __init__(self):
        self.behaviour: dict[str, callable] = {}  # host -> fn(body, attempt) -> (tokens_to_send, send_done | Exception)
        self.agent_ids: dict[str, str] = {}
        self.calls: list[dict] = []
        self.attempts: dict[tuple[str, str], int] = {}

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    async def handle(self, request: httpx.Request) -> httpx.Response:
        host = request.url.host
        raw = request.content
        token = request.headers["authorization"].split(" ", 1)[1]
        claims = jwt.decode(
            token,
            auth.signing_key().public_key,
            algorithms=["RS256"],
            audience=f"aether-seller:{self.agent_ids[host]}",
            issuer=auth.settings.jwt_issuer,
        )
        assert claims["typ"] == "delivery"
        assert claims["body_sha256"] == hashlib.sha256(raw).hexdigest()
        body = json.loads(raw)
        key = (host, body["job_id"])
        attempt = self.attempts.get(key, 0)
        self.attempts[key] = attempt + 1
        self.calls.append({"host": host, "resume_from": body["resume_from"], "attempt": attempt, "claims": claims})
        count, finish = self.behaviour[host](body, attempt)

        async def stream():
            for i in range(body["resume_from"], min(body["resume_from"] + count, body["max_tokens"])):
                yield f"data: {json.dumps({'index': i, 'token': f' t{i}'})}\n\n".encode()
            if isinstance(finish, Exception):
                raise finish
            if finish:
                yield b'data: {"done": true, "finish_reason": "length"}\n\n'

        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=stream())
