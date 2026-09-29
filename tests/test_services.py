import asyncio
import hashlib
import json

import httpx
import jwt
import pytest

from app import auth
from app.config import settings

from .helpers import audit_ok, me, new_agent, new_seller


class FakeAgents:
    """Seller agent endpoints behind an httpx MockTransport; each host has a behaviour."""

    def __init__(self):
        self.behaviour = {}
        self.agent_ids = {}
        self.seen = []

    async def handle(self, request: httpx.Request) -> httpx.Response:
        host = request.url.host
        token = request.headers["authorization"].split(" ", 1)[1]
        claims = jwt.decode(
            token,
            auth.signing_key().public_key,
            algorithms=["RS256"],
            audience=f"aether-seller:{self.agent_ids[host]}",
            issuer=settings.jwt_issuer,
        )
        assert claims["typ"] == "service_call" and claims["body_sha256"] == hashlib.sha256(request.content).hexdigest()
        body = json.loads(request.content)
        assert claims["sub"] == body["call_id"]
        self.seen.append(body)
        return await self.behaviour[host](body)


@pytest.fixture
async def agents(app):
    fake = FakeAgents()
    original = app.state.http
    app.state.http = httpx.AsyncClient(transport=httpx.MockTransport(fake.handle))
    yield fake
    await app.state.http.aclose()
    app.state.http = original


async def list_service(client, agents, host, behaviour, price="0.05", **extra):
    seller = await new_seller(client, host, f"http://{host}/task")
    agents.agent_ids[host], agents.behaviour[host] = seller["agent_id"], behaviour
    r = await client.post(
        "/v1/services",
        headers=seller["headers"],
        json={
            "name": f"Summarize {host}",
            "description": "Summarizes any text in three bullet points.",
            "category": "text",
            "tags": ["summary", "text"],
            "price_usd": price,
            "endpoint_url": f"http://{host}/task",
            "example_input": {"text": "..."},
        }
        | extra,
    )
    assert r.status_code == 201, r.text
    return seller, r.json()


async def ok(body):
    return httpx.Response(200, json={"output": {"summary": body["input"]["text"][:10]}})


async def test_successful_call_pays_seller_minus_fee(client, agents):
    seller, svc = await list_service(client, agents, "sum.test", ok, price="0.05")
    buyer = await new_agent(client, "hirer", ["buy_inference"], fund_usd="1")

    r = await client.post(
        f"/v1/services/{svc['service_id']}/invoke", headers=buyer["headers"], json={"input": {"text": "hello world, long text"}}
    )
    assert r.status_code == 200, r.text
    assert r.json()["output"] == {"summary": "hello worl"} and r.json()["charged_usd"] == "$0.050000000"

    price = 50_000_000
    fee = price * settings.service_fee_bps // 10_000
    assert (await me(client, buyer))["available_nanos"] == 1_000_000_000 - price
    assert (await me(client, seller))["available_nanos"] == price - fee

    listing = (await client.get(f"/v1/services/{svc['service_id']}")).json()
    assert listing["calls"] == 1 and listing["success_rate"] == 1.0
    catalogue = (await client.get("/v1/services?q=summary")).json()["services"]
    assert svc["service_id"] in [s["service_id"] for s in catalogue]
    await audit_ok(client)


async def test_failures_and_timeouts_are_refunded(client, agents, monkeypatch):
    async def broken(body):
        return httpx.Response(500, text="boom")

    async def no_output(body):
        return httpx.Response(200, json={"result": "wrong shape"})

    async def slow(body):
        raise httpx.ReadTimeout("slow")

    buyer = await new_agent(client, "hirer", ["buy_inference"], fund_usd="1")
    for host, behaviour, reason in [
        ("broken.test", broken, "HTTP 500"),
        ("shape.test", no_output, 'missing "output"'),
        ("slow.test", slow, "timed out"),
    ]:
        _, svc = await list_service(client, agents, host, behaviour)
        r = await client.post(f"/v1/services/{svc['service_id']}/invoke", headers=buyer["headers"], json={"input": {}})
        assert r.status_code == 502 and reason in r.json()["detail"]["error_description"]
        assert r.json()["detail"]["charged_usd"] == "$0.000000000"
    wallet = await me(client, buyer)
    assert (wallet["available_nanos"], wallet["escrow_nanos"]) == (1_000_000_000, 0)
    calls = (await client.get("/v1/services/calls/mine", headers=buyer["headers"])).json()
    assert [c["status"] for c in calls] == ["refunded"] * 3
    await audit_ok(client)


async def test_insufficient_funds_self_dealing_price_cap_and_scopes(client, agents):
    seller, svc = await list_service(client, agents, "pricey.test", ok, price="5.00")
    poor = await new_agent(client, "poor", ["buy_inference"])
    r = await client.post(f"/v1/services/{svc['service_id']}/invoke", headers=poor["headers"], json={"input": {}})
    assert r.status_code == 402 and r.json()["detail"]["error"] == "insufficient_funds"

    rich = await new_agent(client, "rich", ["buy_inference"], fund_usd="10")
    r = await client.post(
        f"/v1/services/{svc['service_id']}/invoke", headers=rich["headers"], json={"input": {}, "max_price_usd": "1.00"}
    )
    assert r.status_code == 409  # listed price above the buyer's cap

    both = await new_agent(client, "both", ["buy_inference", "sell_compute"], fund_usd="10")
    await client.put("/v1/agents/me/endpoint", headers=both["headers"], json={"endpoint_url": "http://self.test/t"})
    mine = await client.post(
        "/v1/services",
        headers=both["headers"],
        json={"name": "Self", "description": "self dealing test", "price_usd": "0.01", "endpoint_url": "http://self.test/t"},
    )
    r = await client.post(f"/v1/services/{mine.json()['service_id']}/invoke", headers=both["headers"], json={"input": {}})
    assert r.status_code == 409  # no buying your own service (fake volume/ratings)

    buyer_only = await new_agent(client, "b", ["buy_inference"])
    r = await client.post(
        "/v1/services",
        headers=buyer_only["headers"],
        json={"name": "Nope", "description": "needs sell scope", "price_usd": "0.01", "endpoint_url": "http://x.test/t"},
    )
    assert r.status_code == 403
    await audit_ok(client)


async def test_ratings_pause_and_price_update(client, agents):
    seller, svc = await list_service(client, agents, "rated.test", ok, price="0.02")
    buyer = await new_agent(client, "rater", ["buy_inference"], fund_usd="1")
    call = (
        await client.post(f"/v1/services/{svc['service_id']}/invoke", headers=buyer["headers"], json={"input": {"text": "x"}})
    ).json()
    url = f"/v1/services/calls/{call['call_id']}/rating"
    assert (await client.post(url, headers=buyer["headers"], json={"stars": 4})).status_code == 200
    assert (await client.post(url, headers=buyer["headers"], json={"stars": 5})).status_code == 409
    assert (await client.get(f"/v1/services/{svc['service_id']}")).json()["rating"] == 4.0

    patched = await client.patch(
        f"/v1/services/{svc['service_id']}", headers=seller["headers"], json={"price_usd": "0.03", "status": "paused"}
    )
    assert patched.json()["price_usd"] == "$0.030000000" and patched.json()["status"] == "paused"
    r = await client.post(f"/v1/services/{svc['service_id']}/invoke", headers=buyer["headers"], json={"input": {}})
    assert r.status_code == 404  # paused services can't be bought


async def test_stale_pending_calls_are_refunded(client, agents, monkeypatch):
    from app import services

    release = asyncio.Event()

    async def hangs(body):
        await release.wait()
        return httpx.Response(200, json={"output": "late"})

    _, svc = await list_service(client, agents, "hang.test", hangs)
    buyer = await new_agent(client, "waiter", ["buy_inference"], fund_usd="1")
    task = asyncio.create_task(
        client.post(f"/v1/services/{svc['service_id']}/invoke", headers=buyer["headers"], json={"input": {}})
    )
    for _ in range(50):
        await asyncio.sleep(0.02)
        if agents.seen:
            break
    assert (await me(client, buyer))["escrow_nanos"] == 50_000_000  # escrowed while in flight

    monkeypatch.setattr(settings, "service_max_timeout_s", -120)  # everything pending is now "stale"
    assert await services.refund_stale_calls() == 1
    release.set()
    r = await task  # the late success must not charge again
    assert (await me(client, buyer))["available_nanos"] == 1_000_000_000
    assert r.status_code == 200 and r.json()["charged_usd"] == "$0.000000000"
    await audit_ok(client)
