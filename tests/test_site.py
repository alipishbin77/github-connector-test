from app.config import settings


async def test_landing_page_and_llms_txt_reflect_live_settings(client, monkeypatch):
    monkeypatch.setattr(settings, "public_base_url", "https://aether.example")
    monkeypatch.setattr(settings, "crypto_treasury_address", "0x1454Ad4A90ce0c76b70b61004e0a50E6bA33e36A")
    monkeypatch.setattr(settings, "crypto_chain_name", "Ethereum")
    monkeypatch.setattr(settings, "crypto_chain_id", 1)

    page = await client.get("/")
    assert page.status_code == 200 and page.headers["content-type"].startswith("text/html")
    assert "https://aether.example/v1" in page.text and "USDC on Ethereum" in page.text

    txt = await client.get("/llms.txt")
    assert txt.status_code == 200 and txt.headers["content-type"].startswith("text/plain")
    assert "base_url=https://aether.example/v1" in txt.text
    assert "0x1454Ad4A90ce0c76b70b61004e0a50E6bA33e36A" in txt.text and "/v1/feedback" in txt.text


async def test_landing_page_without_payments(client, monkeypatch):
    monkeypatch.setattr(settings, "crypto_treasury_address", None)
    monkeypatch.setattr(settings, "stripe_secret_key", None)
    assert "not enabled yet" in (await client.get("/llms.txt")).text


async def test_agent_card_and_feedback(client, monkeypatch):
    from .helpers import new_agent

    monkeypatch.setattr(settings, "admin_token", "admin-token-for-feedback-test-123")
    card = (await client.get("/.well-known/agent.json")).json()
    assert card["name"] == "Aether" and card["endpoints"]["feedback"].endswith("/v1/feedback")
    assert (await client.get("/.well-known/agent-card.json")).json() == card

    r = await client.post("/v1/feedback", json={"category": "feature", "message": "please add Solana"})
    assert r.status_code == 201 and r.json()["linked_agent"] is False
    agent = await new_agent(client, "fb", ["buy_inference"])
    r = await client.post("/v1/feedback", headers=agent["headers"], json={"category": "bug", "message": "x broke"})
    assert r.json()["linked_agent"] is True
    assert (await client.post("/v1/feedback", json={"category": "spam!", "message": "hi there"})).status_code == 422

    assert (await client.get("/v1/admin/feedback")).status_code == 403
    rows = (await client.get("/v1/admin/feedback", headers={"X-Admin-Token": "admin-token-for-feedback-test-123"})).json()
    assert [r["message"] for r in rows[:2]] == ["x broke", "please add Solana"] and rows[0]["agent_id"] == agent["agent_id"]


def test_missing_columns_are_added_to_existing_tables(tmp_path):
    from sqlalchemy import create_engine, inspect

    from app.db import _add_missing_columns

    engine = create_engine(f"sqlite:///{tmp_path}/old.db")
    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE crypto_payouts (id VARCHAR PRIMARY KEY)")
        _add_missing_columns(conn)
        _add_missing_columns(conn)  # idempotent
    assert "chain_id" in {c["name"] for c in inspect(engine).get_columns("crypto_payouts")}
