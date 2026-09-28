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
    assert "0x1454Ad4A90ce0c76b70b61004e0a50E6bA33e36A" in txt.text and "chain id 1" in txt.text


async def test_landing_page_without_payments(client, monkeypatch):
    monkeypatch.setattr(settings, "crypto_treasury_address", None)
    monkeypatch.setattr(settings, "stripe_secret_key", None)
    assert "not enabled yet" in (await client.get("/llms.txt")).text
