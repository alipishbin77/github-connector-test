import asyncio

import httpx
import pytest

from .helpers import MockSellers, ask, audit_ok, bid, me, new_agent, new_seller, parse_sse


@pytest.fixture
async def sellers(app):
    mock = MockSellers()
    original = app.state.http
    app.state.http = httpx.AsyncClient(transport=mock.transport())
    yield mock
    await app.state.http.aclose()
    app.state.http = original


async def market(client, sellers, instrument, specs, buy_tokens, cap="1.00"):
    """specs: list of (host, price, capacity, behaviour)."""
    out = []
    for host, price, capacity, behaviour in specs:
        s = await new_seller(client, host, f"http://{host}/gen")
        sellers.agent_ids[host] = s["agent_id"]
        sellers.behaviour[host] = behaviour
        await ask(client, s, instrument, price, capacity)
        out.append(s)
    buyer = await new_agent(client, "buyer", ["buy_inference"], fund_usd="1")
    r = await bid(client, buyer, instrument, cap, buy_tokens, order_type="market")
    assert r.status_code == 201, r.text
    return buyer, out


def healthy(body, attempt):
    return 10**9, True


async def infer(client, buyer, instrument, max_tokens, **extra):
    r = await client.post(
        "/v1/inference",
        headers=buyer["headers"],
        json={"instrument": instrument, "prompt": "hello", "max_tokens": max_tokens} | extra,
    )
    return r, parse_sse(r.text) if r.status_code == 200 else None


async def test_requires_escrowed_allocation(client, sellers, instrument):
    buyer = await new_agent(client, "b", ["buy_inference"], fund_usd="1")
    r, _ = await infer(client, buyer, instrument, 10)
    assert r.status_code == 402 and r.json()["detail"]["error"] == "no_escrowed_allocation"


async def test_stream_and_exact_settlement(client, sellers, instrument):
    buyer, (seller,) = await market(client, sellers, instrument, [("a.test", "0.35", 1000, healthy)], 1000)
    r, events = await infer(client, buyer, instrument, 40)
    tokens = [d for e, d in events if e is None]
    assert [t["index"] for t in tokens] == list(range(40))
    done = events[-1]
    assert done[0] == "done" and done[1]["tokens"] == 40 and done[1]["cost_usd"] == "$0.000014000"

    s = await me(client, seller)
    assert s["available_nanos"] == 40 * 350 - (40 * 350 * 100 // 10_000)  # minus 1% fee
    trade = (await client.get("/v1/trades", headers=buyer["headers"])).json()[0]
    assert (trade["tokens_used"], trade["tokens_reserved"]) == (40, 0)
    assert sellers.calls[0]["claims"]["max_tokens"] == 40
    await audit_ok(client)


async def test_preemption_resumes_from_checkpoint(client, sellers, instrument):
    def flaky(body, attempt):  # first attempt: 7 tokens then the instance disappears
        return (7, None) if attempt == 0 else (10**9, True)

    buyer, _ = await market(client, sellers, instrument, [("a.test", "0.35", 1000, flaky)], 1000)
    r, events = await infer(client, buyer, instrument, 30)
    assert [d["index"] for e, d in events if e is None] == list(range(30))
    resumes = [d for e, d in events if e == "resume"]
    assert len(resumes) == 1 and resumes[0]["resume_from"] == 7 and not resumes[0]["failover"]
    assert [c["resume_from"] for c in sellers.calls] == [0, 7]
    assert events[-1][1]["attempts"] == 2
    await audit_ok(client)


async def test_transport_errors_fail_over_and_split_payment(client, sellers, instrument):
    def dying(body, attempt):
        return 5, httpx.RemoteProtocolError("peer closed connection without sending complete message body")

    buyer, (a, b) = await market(
        client, sellers, instrument, [("a.test", "0.30", 100, dying), ("b.test", "0.40", 100, healthy)], 200
    )
    r, events = await infer(client, buyer, instrument, 50)
    assert [d["index"] for e, d in events if e is None] == list(range(50))
    done = events[-1][1]
    assert events[-1][0] == "done" and done["failovers"] == 1
    assert [s["tokens"] for s in done["segments"]] == [10, 40]  # 5 + 5 from a, the rest from b
    assert [(c["host"], c["resume_from"]) for c in sellers.calls] == [("a.test", 0), ("a.test", 5), ("b.test", 10)]
    gross_a, gross_b = 10 * 300, 40 * 400
    assert (await me(client, a))["available_nanos"] == gross_a - gross_a // 100
    assert (await me(client, b))["available_nanos"] == gross_b - gross_b // 100
    await audit_ok(client)


async def test_unrecoverable_failure_charges_only_delivered_tokens(client, sellers, instrument):
    def dying(body, attempt):
        return 3, httpx.ReadTimeout("seller stalled")

    buyer, _ = await market(client, sellers, instrument, [("a.test", "0.50", 100, dying)], 100)
    before = await me(client, buyer)
    r, events = await infer(client, buyer, instrument, 20)
    assert events[-1][0] == "error" and events[-1][1]["tokens"] == 6
    after = await me(client, buyer)
    assert before["escrow_nanos"] - after["escrow_nanos"] == 6 * 500
    trade = (await client.get("/v1/trades", headers=buyer["headers"])).json()[0]
    assert (trade["tokens_used"], trade["tokens_reserved"], trade["status"]) == (6, 0, "active")

    job_id = events[0][1]["job_id"]
    job = (await client.get(f"/v1/jobs/{job_id}", headers=buyer["headers"])).json()
    assert job["status"] == "failed" and job["checkpointed_text"] == "".join(f" t{i}" for i in range(6))

    released = (await client.post(f"/v1/trades/{trade['trade_id']}/release", headers=buyer["headers"])).json()
    assert released["status"] == "released"
    assert (await me(client, buyer))["escrow_nanos"] == 0
    await audit_ok(client)


async def test_concurrent_requests_cannot_overdraw_an_allocation(client, sellers, instrument):
    buyer, _ = await market(client, sellers, instrument, [("a.test", "0.35", 100, healthy)], 100)
    results = await asyncio.gather(*(infer(client, buyer, instrument, 40) for _ in range(3)))
    codes = sorted(r.status_code for r, _ in results)
    assert codes == [200, 200, 402]
    trade = (await client.get("/v1/trades", headers=buyer["headers"])).json()[0]
    assert (trade["tokens_used"], trade["tokens_reserved"]) == (80, 0)
    await audit_ok(client)


async def test_orphaned_job_is_settled_from_checkpoint(app, client, sellers, instrument):
    from app.db import SessionLocal
    from app.exchange import reserve_allocation
    from app.models import InferenceJob
    from app.proxy_router import Checkpointer, recover_orphaned_jobs

    buyer, (seller,) = await market(client, sellers, instrument, [("a.test", "0.35", 100, healthy)], 100)
    # Simulate a proxy process that reserved capacity, streamed 5 tokens, then died.
    async with SessionLocal() as session, session.begin():
        session.add(
            InferenceJob(
                id="job_orphan_" + instrument,
                buyer_id=buyer["agent_id"],
                instrument=instrument,
                tokens_requested=50,
                status="streaming",
                tokens_delivered=0,
                cost_nanos=0,
                attempts=1,
                failovers=0,
            )
        )
    seg = await reserve_allocation(buyer_id=buyer["agent_id"], instrument=instrument, tokens=50)
    ckpt = Checkpointer(app.state.redis, "job_orphan_" + instrument)
    await ckpt.save_segments([seg])
    for i in range(5):
        await ckpt.add(i, f" t{i}", seg.trade_id)
    await ckpt.flush()
    await app.state.redis.delete(ckpt.heartbeat_key)  # process gone: heartbeat lapses

    assert await recover_orphaned_jobs(app.state.redis, min_age_s=0) >= 1
    trade = (await client.get("/v1/trades", headers=buyer["headers"])).json()[0]
    assert (trade["tokens_used"], trade["tokens_reserved"]) == (5, 0)
    assert (await me(client, seller))["available_nanos"] == 5 * 350 - (5 * 350) // 100
    await audit_ok(client)
