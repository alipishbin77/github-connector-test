import json

import httpx
import pytest

from app.config import settings

from .helpers import MockSellers, ask, audit_ok, me, new_agent, new_seller


@pytest.fixture
async def sellers(app):
    mock = MockSellers()
    original = app.state.http
    app.state.http = httpx.AsyncClient(transport=mock.transport())
    yield mock
    await app.state.http.aclose()
    app.state.http = original


def healthy(body, attempt):
    return 10**9, True


async def listed(client, sellers, instrument, specs):
    out = []
    for host, price, capacity, behaviour in specs:
        s = await new_seller(client, host, f"http://{host}/gen")
        sellers.agent_ids[host] = s["agent_id"]
        sellers.behaviour[host] = behaviour
        await ask(client, s, instrument, price, capacity)
        out.append(s)
    return out


def key_headers(agent):
    return {"Authorization": f"Bearer {agent['api_key']}"}


async def buyer_with_key(client, fund="1"):
    reg = (await client.post("/v1/agents", json={"name": "oa", "scopes": ["buy_inference"]})).json()
    (await client.post("/v1/sandbox/faucet", headers=key_headers(reg), json={"amount_usd": fund})).raise_for_status()
    return reg | {"headers": key_headers(reg)}


async def test_chat_completion_auto_buys_and_rolls_over_allocations(client, sellers, instrument):
    a, b = await listed(client, sellers, instrument, [("a.test", "0.30", 30, healthy), ("b.test", "0.40", 1000, healthy)])
    buyer = await buyer_with_key(client)
    r = await client.post(
        "/v1/chat/completions",
        headers=buyer["headers"],
        json={
            "model": instrument,
            "messages": [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}],
            "max_tokens": 50,
            "max_price_usd_per_mtok": "0.50",
            "temperature": 0.2,  # unsupported OpenAI params are tolerated
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["object"] == "chat.completion" and body["model"] == instrument
    assert body["choices"][0]["message"]["content"] == "".join(f" t{i}" for i in range(50))
    assert body["usage"]["completion_tokens"] == 50
    # 30 tokens from the cheap allocation, then a rollover to the next one.
    assert [s["tokens"] for s in body["aether"]["segments"]] == [30, 20]
    assert [(c["host"], c["resume_from"], c["claims"]["max_tokens"]) for c in sellers.calls] == [
        ("a.test", 0, 30),
        ("b.test", 30, 50),
    ]
    assert sellers.calls[0]["claims"]["body_sha256"]  # structured messages are covered by the body hash
    assert body["aether"]["cost_usd"] == "$0.000017000"  # 30*300 + 20*400 nano-USD
    await audit_ok(client)


async def test_streaming_format_and_usage(client, sellers, instrument):
    await listed(client, sellers, instrument, [("a.test", "0.30", 1000, healthy)])
    buyer = await buyer_with_key(client)
    r = await client.post(
        "/v1/chat/completions",
        headers=buyer["headers"],
        json={
            "model": instrument,
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            "max_completion_tokens": 5,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    )
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    lines = [line[6:] for line in r.text.splitlines() if line.startswith("data: ")]
    assert lines[-1] == "[DONE]"
    chunks = [json.loads(line) for line in lines[:-1]]
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    content = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks if c["choices"])
    assert content == "".join(f" t{i}" for i in range(5))
    assert chunks[-2]["choices"][0]["finish_reason"] == "length"
    assert chunks[-1]["usage"]["completion_tokens"] == 5 and chunks[-1]["choices"] == []


async def test_models_and_quote(client, sellers, instrument):
    await listed(client, sellers, instrument, [("a.test", "0.30", 100, healthy), ("b.test", "0.50", 100, healthy)])
    models = (await client.get("/v1/models")).json()
    entry = next(m for m in models["data"] if m["id"] == instrument)
    assert entry["best_ask_usd_per_mtok"] == "0.3" and entry["ask_depth_tokens"] == 200
    q = (await client.get(f"/v1/quote/{instrument}?tokens=150")).json()
    assert q["fillable_tokens"] == 150 and q["estimated_cost_usd"] == "$0.000055000"
    assert q["average_price_usd_per_mtok"] == "0.366667"


async def test_insufficient_liquidity_is_refunded(client, sellers, instrument):
    await listed(client, sellers, instrument, [("a.test", "0.90", 1000, healthy)])
    buyer = await buyer_with_key(client)
    r = await client.post(
        "/v1/chat/completions",
        headers=buyer["headers"],
        json={
            "model": instrument,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 10,
            "max_price_usd_per_mtok": "0.50",
        },
    )
    assert r.status_code == 503 and r.json()["detail"]["error"]["code"] == "insufficient_liquidity"
    wallet = await me(client, buyer)
    assert wallet["escrow_nanos"] == 0 and wallet["available_nanos"] == 1_000_000_000
    await audit_ok(client)


async def test_auto_buy_disabled_returns_402(client, sellers, instrument):
    buyer = await buyer_with_key(client)
    r = await client.post(
        "/v1/chat/completions",
        headers=buyer["headers"],
        json={"model": instrument, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 10, "auto_buy": False},
    )
    assert r.status_code == 402 and r.json()["detail"]["error"]["code"] == "insufficient_quota"


async def test_api_key_rotation_revokes_old_credentials(client):
    reg = (await client.post("/v1/agents", json={"name": "k", "scopes": ["buy_inference"]})).json()
    assert (await client.get("/v1/agents/me", headers=key_headers(reg))).status_code == 200
    assert (await client.get("/v1/agents/me", headers={"Authorization": f"Bearer {reg['api_key']}x"})).status_code == 401
    rotated = (await client.post("/v1/agents/me/rotate-secret", headers=key_headers(reg))).json()
    assert (await client.get("/v1/agents/me", headers=key_headers(reg))).status_code == 401
    assert (await client.get("/v1/agents/me", headers=key_headers(rotated))).status_code == 200


async def test_rate_limit(client, monkeypatch):
    agent = await new_agent(client, "rl", ["buy_inference"])
    monkeypatch.setattr(settings, "rate_limit_per_minute", 3)
    codes = [(await client.get("/v1/agents/me", headers=agent["headers"])).status_code for _ in range(5)]
    assert codes.count(429) >= 1 and codes[0] == 200
