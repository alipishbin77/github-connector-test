import asyncio

from app.units import usd_per_mtok_to_npt

from .helpers import ask, audit_ok, bid, me, new_agent, new_seller


def test_price_units():
    assert usd_per_mtok_to_npt("0.35") == 350
    assert usd_per_mtok_to_npt("25") == 25_000
    for bad in ("0.0005", "0", "-1"):
        try:
            usd_per_mtok_to_npt(bad)
        except ValueError:
            continue
        raise AssertionError(bad)


async def test_price_time_priority_and_price_improvement(client, instrument):
    s1 = await new_seller(client, "s1", "http://s1.test/gen")
    s2 = await new_seller(client, "s2", "http://s2.test/gen")
    s3 = await new_seller(client, "s3", "http://s3.test/gen")
    await ask(client, s1, instrument, "0.40", 100)  # worse price, earliest
    await ask(client, s2, instrument, "0.35", 100)  # best price, earlier
    await ask(client, s3, instrument, "0.35", 100)  # best price, later
    buyer = await new_agent(client, "b", ["buy_inference"], fund_usd="1")

    r = await bid(client, buyer, instrument, "0.40", 250, time_in_force="ioc")
    assert r.status_code == 201, r.text
    fills = r.json()["fills"]
    assert [(f["counterparty_id"], f["tokens_total"], f["price_usd_per_mtok"]) for f in fills] == [
        (s2["agent_id"], 100, "0.35"),
        (s3["agent_id"], 100, "0.35"),
        (s1["agent_id"], 50, "0.4"),
    ]
    assert r.json()["order"]["status"] == "filled"
    wallet = await me(client, buyer)
    # escrow = 100*350 + 100*350 + 50*400; the cap (400) minus 350 on 200 tokens was refunded
    assert wallet["escrow_nanos"] == 90_000
    assert wallet["available_nanos"] == 1_000_000_000 - 90_000

    book = (await client.get(f"/v1/book/{instrument}")).json()
    assert book["asks"] == [{"price_usd_per_mtok": "0.4", "tokens": 50, "orders": 1}]
    await audit_ok(client)


async def test_resting_bid_is_hit_by_incoming_ask_then_cancelled(client, instrument):
    buyer = await new_agent(client, "b", ["buy_inference"], fund_usd="1")
    r = await bid(client, buyer, instrument, "0.50", 100)
    assert r.json()["order"]["status"] == "open" and r.json()["fills"] == []
    order_id = r.json()["order"]["order_id"]
    assert (await me(client, buyer))["escrow_nanos"] == 50_000

    seller = await new_seller(client, "s", "http://s.test/gen")
    placed = await ask(client, seller, instrument, "0.30", 60)
    # The ask is the taker: it executes at the resting bid's price.
    assert [(f["tokens_total"], f["price_usd_per_mtok"]) for f in placed["fills"]] == [(60, "0.5")]

    order = (await client.get(f"/v1/orders/{order_id}", headers=buyer["headers"])).json()
    assert order["filled_tokens"] == 60 and order["open_tokens"] == 40 and order["status"] == "open"

    cancelled = (await client.delete(f"/v1/orders/{order_id}", headers=buyer["headers"])).json()
    assert cancelled["status"] == "cancelled" and cancelled["cancelled_tokens"] == 40
    wallet = await me(client, buyer)
    assert wallet["escrow_nanos"] == 60 * 500  # only the allocation stays escrowed
    assert (await client.delete(f"/v1/orders/{order_id}", headers=buyer["headers"])).status_code == 409
    await audit_ok(client)


async def test_self_trade_prevention(client, instrument):
    both = await new_agent(client, "mm", ["buy_inference", "sell_compute"], fund_usd="1")
    await client.put("/v1/agents/me/endpoint", headers=both["headers"], json={"endpoint_url": "http://mm.test/gen"})
    await ask(client, both, instrument, "0.30", 100)
    r = await bid(client, both, instrument, "0.50", 100, time_in_force="ioc")
    assert r.json()["fills"] == [] and r.json()["order"]["status"] == "cancelled"
    book = (await client.get(f"/v1/book/{instrument}")).json()
    assert book["asks"][0]["tokens"] == 100
    await audit_ok(client)


async def test_market_order_ioc_remainder_is_refunded(client, instrument):
    seller = await new_seller(client, "s", "http://s.test/gen")
    await ask(client, seller, instrument, "0.20", 100)
    buyer = await new_agent(client, "b", ["buy_inference"], fund_usd="1")
    r = await bid(client, buyer, instrument, "0.50", 500, order_type="market")
    order = r.json()["order"]
    assert (order["filled_tokens"], order["cancelled_tokens"], order["status"]) == (100, 400, "cancelled")
    assert (await me(client, buyer))["escrow_nanos"] == 100 * 200
    await audit_ok(client)


async def test_insufficient_funds_and_scope(client, instrument):
    buyer = await new_agent(client, "poor", ["buy_inference"], fund_usd="0.000001")
    r = await bid(client, buyer, instrument, "1.00", 10_000)
    assert r.status_code == 402
    r = await client.post(
        "/v1/orders",
        headers=buyer["headers"],
        json={"instrument": instrument, "side": "ask", "price_usd_per_mtok": "1", "quantity_tokens": 1},
    )
    assert r.status_code == 403 and r.json()["detail"]["error"] == "insufficient_scope"
    r = await bid(client, buyer, instrument, "0.0005", 1)
    assert r.status_code == 422


async def test_resting_ask_expires(client, instrument):
    seller = await new_seller(client, "s", "http://s.test/gen")
    placed = await ask(client, seller, instrument, "0.30", 100, ttl_seconds=1)
    order_id = placed["order"]["order_id"]
    for _ in range(40):
        await asyncio.sleep(0.1)
        order = (await client.get(f"/v1/orders/{order_id}", headers=seller["headers"])).json()
        if order["status"] == "expired":
            break
    assert order["status"] == "expired" and order["cancelled_tokens"] == 100
    assert (await client.get(f"/v1/book/{instrument}")).json()["asks"] == []
