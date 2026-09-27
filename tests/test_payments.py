import hashlib
import hmac
import json
import time
from urllib.parse import parse_qs

import httpx
import pytest

from app import payments
from app.config import settings

from .helpers import audit_ok, me, new_agent, new_seller

WEBHOOK_SECRET = "whsec_test_secret"


class FakeStripe:
    def __init__(self):
        self.calls: list[tuple[str, dict, dict]] = []
        self.fail_transfers = False

    async def handle(self, request: httpx.Request) -> httpx.Response:
        form = {k: v[0] for k, v in parse_qs(request.content.decode()).items()}
        self.calls.append((request.url.path, form, dict(request.headers)))
        path = request.url.path
        if path == "/v1/checkout/sessions":
            return httpx.Response(200, json={"id": f"cs_test_{len(self.calls)}", "url": "https://checkout.stripe.test/pay"})
        if path == "/v1/accounts":
            return httpx.Response(200, json={"id": "acct_test_1"})
        if path == "/v1/account_links":
            return httpx.Response(200, json={"url": "https://connect.stripe.test/onboard"})
        if path == "/v1/transfers":
            if self.fail_transfers:
                return httpx.Response(400, json={"error": {"message": "insufficient platform balance"}})
            return httpx.Response(200, json={"id": "tr_test_1"})
        return httpx.Response(404, json={"error": {"message": "unknown"}})


@pytest.fixture
async def stripe(app, monkeypatch):
    fake = FakeStripe()
    client = payments.StripeClient(
        httpx.AsyncClient(base_url="https://api.stripe.test", transport=httpx.MockTransport(fake.handle))
    )
    monkeypatch.setattr(app.state, "stripe", client, raising=False)
    monkeypatch.setattr(settings, "stripe_webhook_secret", WEBHOOK_SECRET)
    yield fake
    await client.aclose()


def signed(payload: dict, secret: str = WEBHOOK_SECRET, ts: int | None = None) -> tuple[bytes, str]:
    body = json.dumps(payload).encode()
    ts = ts or int(time.time())
    sig = hmac.new(secret.encode(), f"{ts}.".encode() + body, hashlib.sha256).hexdigest()
    return body, f"t={ts},v1={sig}"


def paid_event(session_id: str, agent_id: str, cents: int) -> dict:
    return {
        "type": "checkout.session.completed",
        "data": {
            "object": {
                "id": session_id,
                "payment_status": "paid",
                "currency": "usd",
                "amount_total": cents,
                "metadata": {"agent_id": agent_id},
            }
        },
    }


async def test_deposit_checkout_and_idempotent_webhook(client, stripe):
    agent = await new_agent(client, "payer", ["buy_inference"])
    r = await client.post("/v1/billing/deposits", headers=agent["headers"], json={"amount_usd": "25.00"})
    assert r.status_code == 200 and r.json()["checkout_url"].startswith("https://checkout.stripe.test")
    path, form, headers = stripe.calls[-1]
    assert form["line_items[0][price_data][unit_amount]"] == "2500" and form["metadata[agent_id]"] == agent["agent_id"]
    assert headers["idempotency-key"]

    body, sig = signed(paid_event("cs_live_abc", agent["agent_id"], 2500))
    for _ in range(2):  # Stripe retries; the second delivery must not double-credit
        r = await client.post("/v1/billing/stripe/webhook", content=body, headers={"Stripe-Signature": sig})
        assert r.status_code == 200
    assert (await me(client, agent))["available_nanos"] == 25 * 10**9
    await audit_ok(client)


async def test_webhook_rejects_bad_or_stale_signatures(client, stripe):
    agent = await new_agent(client, "payer", ["buy_inference"])
    body, _ = signed(paid_event("cs_forged", agent["agent_id"], 10**6))
    _, bad = signed(paid_event("cs_forged", agent["agent_id"], 10**6), secret="whsec_wrong")
    assert (await client.post("/v1/billing/stripe/webhook", content=body, headers={"Stripe-Signature": bad})).status_code == 400
    body, stale = signed(paid_event("cs_old", agent["agent_id"], 100), ts=int(time.time()) - 3600)
    assert (await client.post("/v1/billing/stripe/webhook", content=body, headers={"Stripe-Signature": stale})).status_code == 400
    assert (await me(client, agent))["available_nanos"] == 0


async def test_seller_onboarding_and_payouts(client, stripe):
    seller = await new_seller(client, "earner", "http://earner.test/gen")
    assert (
        await client.post("/v1/billing/withdrawals", headers=seller["headers"], json={"amount_usd": "10.00"})
    ).status_code == 409

    r = await client.post("/v1/billing/connect/onboard", headers=seller["headers"])
    assert r.json() == {"onboarding_url": "https://connect.stripe.test/onboard", "stripe_account_id": "acct_test_1"}

    # Earnings arrive as available balance (simulated here with a deposit credit).
    await payments.credit_deposit("cs_seed_" + seller["agent_id"], seller["agent_id"], 30 * 10**9)
    assert (
        await client.post("/v1/billing/withdrawals", headers=seller["headers"], json={"amount_usd": "50.00"})
    ).status_code == 402

    r = await client.post("/v1/billing/withdrawals", headers=seller["headers"], json={"amount_usd": "12.34"})
    assert r.status_code == 200 and r.json()["transfer_id"] == "tr_test_1"
    _, form, headers = stripe.calls[-1]
    assert form["amount"] == "1234" and form["destination"] == "acct_test_1"
    assert headers["idempotency-key"] == r.json()["withdrawal_id"]
    assert (await me(client, seller))["available_nanos"] == 30 * 10**9 - 1234 * 10**7

    stripe.fail_transfers = True
    r = await client.post("/v1/billing/withdrawals", headers=seller["headers"], json={"amount_usd": "10.00"})
    assert r.status_code == 502
    assert (await me(client, seller))["available_nanos"] == 30 * 10**9 - 1234 * 10**7  # reversed
    statuses = [w["status"] for w in (await client.get("/v1/billing/withdrawals", headers=seller["headers"])).json()]
    assert sorted(statuses) == ["failed", "paid"]

    buyer_only = await new_agent(client, "b", ["buy_inference"])
    assert (await client.post("/v1/billing/connect/onboard", headers=buyer_only["headers"])).status_code == 403
    stats = (await client.get("/v1/admin/stats")).json()
    assert stats["payouts_usd"] != "$0.000000000"
    await audit_ok(client)


async def test_billing_disabled_without_keys(app, client, monkeypatch):
    monkeypatch.setattr(app.state, "stripe", None, raising=False)
    agent = await new_agent(client, "x", ["buy_inference"])
    assert (await client.post("/v1/billing/deposits", headers=agent["headers"], json={"amount_usd": "10.00"})).status_code == 503
