"""House services: the three tasks the platform sells itself.

Each test calls the seller endpoint the way the clearinghouse does — compact
JSON body, single-use token bound to that body — because that is the only
contract these endpoints have.
"""

import hashlib
import json

import httpx
import pytest
from fastapi import HTTPException

from app import auth, netguard
from app.config import settings
from app.readable import html_to_markdown

from .helpers import new_agent

# Literal addresses, so nothing here needs DNS: getaddrinfo resolves numeric
# hosts locally. 93.184.216.34 is a public address, the rest must be refused.
PUBLIC = "93.184.216.34"
PRIVATE_HOSTS = [
    "127.0.0.1",  # loopback
    "169.254.169.254",  # cloud metadata
    "10.77.0.1",  # the host this container runs on
    "100.114.15.67",  # the host's Tailscale address (CGNAT: not is_private!)
    "[::1]",  # IPv6 loopback
    "192.168.1.1",
]

SLUGS = ("webpage-to-markdown", "json-schema-validate", "usdc-balance")


# ------------------------------------------------------------------- fixtures


class FakeWeb:
    """Stands in for the public internet behind app.state.http."""

    def __init__(self):
        self.responses: dict[str, callable] = {}  # path -> fn(request) -> Response
        self.requested: list[str] = []

    async def handle(self, request: httpx.Request) -> httpx.Response:
        self.requested.append(str(request.url))
        handler = self.responses.get(request.url.path)
        if handler is None:
            return httpx.Response(404, text="no such page")
        return handler(request)


@pytest.fixture
async def web(app):
    fake = FakeWeb()
    original = app.state.http
    app.state.http = httpx.AsyncClient(transport=httpx.MockTransport(fake.handle), follow_redirects=False)
    yield fake
    await app.state.http.aclose()
    app.state.http = original


@pytest.fixture
async def house(client, monkeypatch):
    """One agent listing all three house services, as the operator would. It must
    be registered as `house_agent_id`: only that agent may be paid for work this
    server performs."""
    seller = await new_agent(client, "aether-house", ["sell_compute"])
    monkeypatch.setattr(settings, "house_agent_id", seller["agent_id"])
    services = {}
    for slug in SLUGS:
        r = await client.post(
            "/v1/services",
            headers=seller["headers"],
            json={
                "name": f"House: {slug}",
                "description": "Run by the platform itself. No GPU, no LLM.",
                "category": "web",
                "price_usd": "0.002",
                "endpoint_url": f"http://aether.test/v1/house/{slug}",
            },
        )
        assert r.status_code == 201, r.text
        services[slug] = r.json()["service_id"]
    return {"seller": seller, "services": services}


def call_body(house, slug, payload, call_id="call_house_1"):
    service_id = house["services"][slug]
    raw = json.dumps({"call_id": call_id, "service_id": service_id, "input": payload}, separators=(",", ":")).encode()
    token = auth.issue_service_token(
        seller_id=house["seller"]["agent_id"],
        call_id=call_id,
        service_id=service_id,
        price_nanos=2_000_000,
        body_sha256=hashlib.sha256(raw).hexdigest(),
    )
    return raw, token


async def invoke(client, house, slug, payload, call_id=None, token=None, raw=None):
    """Call a house endpoint exactly as services.invoke does."""
    body, minted = call_body(house, slug, payload, call_id=call_id or f"call_{slug}_{id(payload)}")
    return await client.post(
        f"/v1/house/{slug}",
        content=raw if raw is not None else body,
        headers={"Authorization": f"Bearer {minted if token is None else token}", "Content-Type": "application/json"},
    )


PAGE = """
<html><head><title>Aether</title><style>p{color:red}</style></head>
<body>
  <nav><a href="/elsewhere">skip me</a></nav>
  <main>
    <h1>Agents hire agents</h1>
    <p>Sellers need <strong>no GPU</strong>. The fee is 10%.</p>
    <ul><li>webpage to markdown</li><li>schema validation</li></ul>
    <p>Read the <a href="/llms.txt">docs</a> or <code>POST /v1/services</code>.</p>
    <pre><code>curl -X POST /v1/services</code></pre>
    <p>A paragraph long enough that the main element is clearly the content of
    this page and not an empty wrapper around the real body text.</p>
  </main>
  <footer><p>do not include me</p></footer>
</body></html>
"""


# ------------------------------------------------------------- happy paths


async def test_webpage_to_markdown_returns_readable_markdown(client, house, web):
    web.responses["/page"] = lambda r: httpx.Response(200, headers={"content-type": "text/html; charset=utf-8"}, text=PAGE)

    r = await invoke(client, house, "webpage-to-markdown", {"url": f"http://{PUBLIC}/page"})

    assert r.status_code == 200, r.text
    markdown = r.json()["output"]
    assert "# Agents hire agents" in markdown
    assert "**no GPU**" in markdown
    assert "- webpage to markdown" in markdown
    assert f"[docs](http://{PUBLIC}/llms.txt)" in markdown  # relative links are absolutised
    assert "```" in markdown
    assert "skip me" not in markdown and "do not include me" not in markdown  # nav/footer dropped
    assert "color:red" not in markdown


async def test_json_schema_validate_reports_valid_and_invalid(client, house):
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer", "minimum": 0}},
        "required": ["name"],
    }

    r = await invoke(client, house, "json-schema-validate", {"schema": schema, "data": {"name": "ada", "age": 36}})
    assert r.status_code == 200, r.text
    assert r.json()["output"]["valid"] is True
    assert r.json()["output"]["errors"] == []

    r = await invoke(client, house, "json-schema-validate", {"schema": schema, "data": {"age": -1}}, call_id="c2")
    assert r.status_code == 200, r.text
    out = r.json()["output"]
    assert out["valid"] is False
    assert out["error_count"] == 2
    assert {e["validator"] for e in out["errors"]} == {"required", "minimum"}


async def test_usdc_balance_reads_balance_of(client, house, web):
    calls = []

    def rpc(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        calls.append(payload)
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": payload["id"], "result": hex(12_345_678)})

    web.responses["/"] = rpc  # base-rpc.publicnode.com/
    address = "0x1454Ad4A90ce0c76b70b61004e0a50E6bA33e36A"

    r = await invoke(client, house, "usdc-balance", {"address": address, "network": "base"})

    assert r.status_code == 200, r.text
    out = r.json()["output"]
    assert out["balance"] == "12.345678"
    assert out["balance_units"] == 12_345_678
    assert (out["network"], out["chain_id"], out["decimals"]) == ("base", 8453, 6)
    assert out["address"] == address.lower()
    assert calls[0]["method"] == "eth_call"
    assert calls[0]["params"][0]["data"].startswith("0x70a08231")  # balanceOf(address)
    assert address[2:].lower() in calls[0]["params"][0]["data"]


async def test_usdc_balance_rpc_failure_refunds_rather_than_crashing(client, house, web):
    web.responses["/"] = lambda r: httpx.Response(500, text="rpc down")

    r = await invoke(client, house, "usdc-balance", {"address": "0x" + "ab" * 20, "network": "arbitrum"})

    assert r.status_code == 502
    assert r.json()["detail"]["error"] == "rpc_failed"


# --------------------------------------------------------------------- auth


@pytest.mark.parametrize("slug", SLUGS)
async def test_missing_token_does_no_work(client, house, web, slug):
    web.responses["/page"] = lambda r: httpx.Response(200, text=PAGE)

    r = await client.post(f"/v1/house/{slug}", json={"call_id": "c", "service_id": house["services"][slug], "input": {}})

    assert r.status_code == 401
    assert web.requested == []


@pytest.mark.parametrize("token", ["not-a-jwt", "", "a.b.c"])
async def test_malformed_token_is_rejected(client, house, web, token):
    web.responses["/page"] = lambda r: httpx.Response(200, text=PAGE)

    r = await invoke(client, house, "webpage-to-markdown", {"url": f"http://{PUBLIC}/page"}, token=token)

    assert r.status_code == 401
    assert web.requested == []


async def test_token_bound_to_a_different_body_is_rejected(client, house, web):
    web.responses["/page"] = lambda r: httpx.Response(200, text=PAGE)
    _, token = call_body(house, "webpage-to-markdown", {"url": f"http://{PUBLIC}/page"})
    tampered = json.dumps(
        {"call_id": "call_x", "service_id": house["services"]["webpage-to-markdown"], "input": {"url": "http://127.0.0.1/"}},
        separators=(",", ":"),
    ).encode()

    r = await client.post("/v1/house/webpage-to-markdown", content=tampered, headers={"Authorization": f"Bearer {token}"})

    assert r.status_code == 401
    assert "does not match" in r.json()["detail"]["error_description"]
    assert web.requested == []


async def test_token_for_one_house_service_cannot_be_spent_on_another(client, house):
    """The audience comes from the listing, and the listing names one endpoint."""
    raw, token = call_body(house, "json-schema-validate", {"schema": {}, "data": 1})

    r = await client.post("/v1/house/usdc-balance", content=raw, headers={"Authorization": f"Bearer {token}"})

    assert r.status_code == 401


async def test_token_for_another_seller_is_rejected(client, house, web):
    other = await new_agent(client, "someone-else", ["sell_compute"])
    service_id = house["services"]["webpage-to-markdown"]
    raw = json.dumps(
        {"call_id": "c9", "service_id": service_id, "input": {"url": f"http://{PUBLIC}/page"}}, separators=(",", ":")
    ).encode()
    token = auth.issue_service_token(
        seller_id=other["agent_id"],  # audience: a different seller
        call_id="c9",
        service_id=service_id,
        price_nanos=2_000_000,
        body_sha256=hashlib.sha256(raw).hexdigest(),
    )

    r = await client.post("/v1/house/webpage-to-markdown", content=raw, headers={"Authorization": f"Bearer {token}"})

    assert r.status_code == 401
    assert web.requested == []


async def test_token_is_single_use(client, house):
    payload = {"schema": {"type": "integer"}, "data": 7}
    raw, token = call_body(house, "json-schema-validate", payload, call_id="replay-me")
    headers = {"Authorization": f"Bearer {token}"}

    first = await client.post("/v1/house/json-schema-validate", content=raw, headers=headers)
    replay = await client.post("/v1/house/json-schema-validate", content=raw, headers=headers)

    assert first.status_code == 200
    assert replay.status_code == 401
    assert replay.json()["detail"]["error_description"] == "token already used"


async def test_access_token_is_not_a_service_call_token(client, house):
    """An agent's own bearer token must not open a seller endpoint."""
    raw, _ = call_body(house, "json-schema-validate", {"schema": True, "data": 1})

    r = await client.post(
        "/v1/house/json-schema-validate", content=raw, headers={"Authorization": f"Bearer {house['seller']['token']}"}
    )

    assert r.status_code == 401


# --------------------------------------------------------------------- SSRF


@pytest.mark.parametrize("host", PRIVATE_HOSTS)
async def test_private_targets_are_refused(client, house, web, host):
    r = await invoke(client, house, "webpage-to-markdown", {"url": f"http://{host}/latest/meta-data/"}, call_id=f"ssrf-{host}")

    assert r.status_code == 422, r.text
    assert r.json()["detail"]["error"] == "refused_url"
    assert web.requested == []  # refused before any connection


async def test_redirect_to_a_private_address_is_refused(client, house, web):
    web.responses["/bounce"] = lambda r: httpx.Response(302, headers={"location": "http://169.254.169.254/latest/meta-data/"})

    r = await invoke(client, house, "webpage-to-markdown", {"url": f"http://{PUBLIC}/bounce"})

    assert r.status_code == 422
    assert r.json()["detail"]["error"] == "refused_url"
    assert web.requested == [f"http://{PUBLIC}/bounce"]  # the metadata service was never contacted


async def test_redirect_chain_is_bounded(client, house, web):
    web.responses["/loop"] = lambda r: httpx.Response(302, headers={"location": f"http://{PUBLIC}/loop"})

    r = await invoke(client, house, "webpage-to-markdown", {"url": f"http://{PUBLIC}/loop"})

    assert r.status_code == 502
    assert r.json()["detail"]["error"] == "too_many_redirects"


@pytest.mark.parametrize("url", ["file:///etc/passwd", "gopher://x/", "not a url", "http://user:pw@" + PUBLIC + "/"])
async def test_non_http_and_credentialed_urls_are_refused(client, house, web, url):
    r = await invoke(client, house, "webpage-to-markdown", {"url": url}, call_id=f"scheme-{url}")

    assert r.status_code == 422, r.text
    assert web.requested == []


async def test_unresolvable_host_is_a_clean_error(client, house, web):
    r = await invoke(client, house, "webpage-to-markdown", {"url": "http://does-not-exist.invalid/page"})

    assert r.status_code == 422
    assert "does not resolve" in r.json()["detail"]["error_description"]


def test_netguard_classifies_addresses():
    import ipaddress

    for host in ["127.0.0.1", "10.77.0.1", "169.254.169.254", "100.114.15.67", "192.0.2.5", "224.0.0.1", "0.0.0.0", "::1"]:
        assert not netguard.ip_is_public(ipaddress.ip_address(host)), host
    for host in ["93.184.216.34", "1.1.1.1", "2606:4700:4700::1111"]:
        assert netguard.ip_is_public(ipaddress.ip_address(host)), host


# ------------------------------------------------------- limits and bad input


async def test_oversized_page_is_refused_without_a_5xx(client, house, web):
    web.responses["/huge"] = lambda r: httpx.Response(200, headers={"content-type": "text/html"}, content=b"<p>x</p>" * 500_000)

    r = await invoke(client, house, "webpage-to-markdown", {"url": f"http://{PUBLIC}/huge"})

    assert r.status_code == 413
    assert r.json()["detail"]["error"] == "page_too_large"


async def test_oversized_page_is_refused_when_only_declared(client, house, web):
    web.responses["/claims-huge"] = lambda r: httpx.Response(
        200, headers={"content-type": "text/html", "content-length": "99000000"}, content=b"<p>small</p>"
    )

    r = await invoke(client, house, "webpage-to-markdown", {"url": f"http://{PUBLIC}/claims-huge"})

    assert r.status_code == 413


async def test_timeout_is_reported_as_504_not_a_crash(client, house, web):
    def timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow", request=request)

    web.responses["/slow"] = timeout

    r = await invoke(client, house, "webpage-to-markdown", {"url": f"http://{PUBLIC}/slow"})

    assert r.status_code == 504
    assert r.json()["detail"]["error"] == "fetch_timeout"


async def test_unreachable_page_is_reported_as_502(client, house, web):
    def refused(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    web.responses["/down"] = refused

    r = await invoke(client, house, "webpage-to-markdown", {"url": f"http://{PUBLIC}/down"})

    assert r.status_code == 502
    assert r.json()["detail"]["error"] == "fetch_failed"


async def test_non_page_content_type_is_refused(client, house, web):
    web.responses["/blob"] = lambda r: httpx.Response(200, headers={"content-type": "application/zip"}, content=b"PK\x03\x04")

    r = await invoke(client, house, "webpage-to-markdown", {"url": f"http://{PUBLIC}/blob"})

    assert r.status_code == 415
    assert r.json()["detail"]["error"] == "not_a_page"


async def test_plain_text_pages_pass_through(client, house, web):
    web.responses["/llms.txt"] = lambda r: httpx.Response(
        200, headers={"content-type": "text/plain; charset=utf-8"}, text="# Aether\n\nAgents hire agents.\n"
    )

    r = await invoke(client, house, "webpage-to-markdown", {"url": f"http://{PUBLIC}/llms.txt"})

    assert r.status_code == 200
    assert r.json()["output"] == "# Aether\n\nAgents hire agents."


@pytest.mark.parametrize(
    ("slug", "payload"),
    [
        ("webpage-to-markdown", {}),
        ("webpage-to-markdown", {"url": 42}),
        ("webpage-to-markdown", "just a string"),
        ("json-schema-validate", {"data": 1}),
        ("json-schema-validate", {"schema": "not a schema", "data": 1}),
        ("usdc-balance", {"address": "nope", "network": "base"}),
        ("usdc-balance", {"address": "0x" + "ab" * 20, "network": "dogecoin"}),
        ("usdc-balance", None),
    ],
)
async def test_bad_input_is_a_clean_error_not_a_stack_trace(client, house, web, slug, payload):
    r = await invoke(client, house, slug, payload, call_id=f"bad-{slug}-{payload}")

    assert r.status_code == 422, r.text
    detail = r.json()["detail"]
    assert set(detail) == {"error", "error_description"}
    assert "Traceback" not in detail["error_description"]
    assert web.requested == []


async def test_invalid_schema_document_is_reported(client, house):
    r = await invoke(client, house, "json-schema-validate", {"schema": {"type": "banana"}, "data": 1})

    assert r.status_code == 422
    assert r.json()["detail"]["error"] == "invalid_schema"


async def test_non_json_body_is_rejected(client, house):
    r = await client.post("/v1/house/json-schema-validate", content=b"<not json>", headers={"Authorization": "Bearer x.y.z"})

    assert r.status_code == 400
    assert r.json()["detail"]["error"] == "invalid_request"


async def test_unknown_service_id_is_rejected(client, house):
    raw = json.dumps({"call_id": "c", "service_id": "svc_nope", "input": {}}, separators=(",", ":")).encode()

    r = await client.post("/v1/house/usdc-balance", content=raw, headers={"Authorization": "Bearer x.y.z"})

    assert r.status_code == 401
    assert r.json()["detail"]["error_description"] == "unknown service_id"


# ------------------------------------------------------- html -> markdown unit


def test_html_to_markdown_handles_tables_quotes_and_broken_markup():
    markdown = html_to_markdown(
        "<h2>T</h2><table><tr><th>a</th><th>b</th></tr><tr><td>1</td><td>2</td></tr></table>"
        "<blockquote>quoted</blockquote><p>unclosed<ul><li>x</li>",
        base_url="https://example.test/",
    )
    assert "## T" in markdown
    assert "| a | b |" in markdown
    assert "| --- | --- |" in markdown
    assert "| 1 | 2 |" in markdown
    assert "> quoted" in markdown
    assert "- x" in markdown


def test_html_to_markdown_output_is_bounded():
    markdown = html_to_markdown("<p>" + "word " * 5000 + "</p>", max_chars=100)
    assert len(markdown) < 200
    assert markdown.endswith("[truncated]")


def test_html_to_markdown_falls_back_when_main_is_empty():
    markdown = html_to_markdown("<main></main><p>The real body text lives outside main on this page.</p>")
    assert "The real body text" in markdown


# ------------------------------------------------------------------- listing

OWN_CGNAT = "100.98.111.112"  # what this host's own Funnel name resolves to locally


@pytest.fixture
def strict_urls(monkeypatch):
    """Production settings: the SSRF guard on, and a public URL that this host
    resolves to a non-public address, which is what Tailscale answers for a
    Funnel name asked from inside the tailnet. A literal CGNAT address stands
    in for that name: a real `*.ts.net` name resolves to a *public* Funnel
    ingress from anywhere else, so asserting on it would pass here and fail on
    a CI runner."""
    monkeypatch.setattr(settings, "allow_private_seller_urls", False)
    monkeypatch.setattr(settings, "public_base_url", f"https://{OWN_CGNAT}")


async def test_house_endpoints_can_be_listed_behind_the_ssrf_guard(client, strict_urls, monkeypatch):
    seller = await new_agent(client, "house-lister", ["sell_compute"])
    monkeypatch.setattr(settings, "house_agent_id", seller["agent_id"])
    listing = {
        "name": "House: webpage to markdown",
        "description": "Run by the platform itself. No GPU, no LLM.",
        "category": "web",
        "price_usd": "0.002",
    }

    own = await client.post(
        "/v1/services",
        headers=seller["headers"],
        json=listing | {"endpoint_url": f"https://{OWN_CGNAT}/v1/house/webpage-to-markdown"},
    )
    other_path = await client.post(
        "/v1/services", headers=seller["headers"], json=listing | {"endpoint_url": f"https://{OWN_CGNAT}/v1/admin"}
    )
    other_origin = await client.post(
        "/v1/services",
        headers=seller["headers"],
        json=listing | {"endpoint_url": "https://10.0.0.5/v1/house/webpage-to-markdown"},
    )

    assert own.status_code == 201, own.text
    # The exemption is one origin and one path prefix: nothing else gets it.
    assert other_path.status_code == 422, other_path.text
    assert other_origin.status_code == 422, other_origin.text


async def test_only_the_house_agent_may_list_a_house_endpoint(client, strict_urls, monkeypatch):
    """Otherwise a stranger lists our own house endpoint as their own service at
    any price and is paid for work this server performs at its own cost — and the
    page fetcher becomes an open web proxy attributable to this host, billed to
    whoever buys. The address exemption must be tied to the house agent."""
    house_agent = await new_agent(client, "house-real", ["sell_compute"])
    stranger = await new_agent(client, "house-impostor", ["sell_compute"])
    monkeypatch.setattr(settings, "house_agent_id", house_agent["agent_id"])
    listing = {
        "name": "Totally my own service",
        "description": "Reselling the platform's own compute as if it were mine.",
        "category": "web",
        "price_usd": "0.002",
        "endpoint_url": f"https://{OWN_CGNAT}/v1/house/webpage-to-markdown",
    }

    assert (await client.post("/v1/services", headers=house_agent["headers"], json=listing)).status_code == 201
    impostor = await client.post("/v1/services", headers=stranger["headers"], json=listing)
    assert impostor.status_code == 422, impostor.text

    # No house agent configured: nobody gets the exemption. Fail closed.
    monkeypatch.setattr(settings, "house_agent_id", None)
    unset = await client.post("/v1/services", headers=house_agent["headers"], json=listing)
    assert unset.status_code == 422, unset.text


async def test_house_call_rejects_a_service_not_owned_by_the_house_agent(client, monkeypatch):
    """Second, independent gate. A listing made before the setting existed, or via
    any future path that skips the listing check, must still not earn here."""
    from app import house

    monkeypatch.setattr(settings, "allow_private_seller_urls", True)  # let the listing itself through
    stranger = await new_agent(client, "house-sneak", ["sell_compute"])
    created = await client.post(
        "/v1/services",
        headers=stranger["headers"],
        json={
            "name": "Sneaky house reseller",
            "description": "Listed against a house path without owning it.",
            "category": "web",
            "price_usd": "0.002",
            "endpoint_url": "https://example.invalid/v1/house/webpage-to-markdown",
        },
    )
    assert created.status_code == 201, created.text

    monkeypatch.setattr(settings, "house_agent_id", "agt_someone_else_entirely")
    with pytest.raises(HTTPException) as caught:
        await house._house_seller(created.json()["service_id"], "webpage-to-markdown")
    assert caught.value.status_code == 401
    assert "not owned by the house agent" in str(caught.value.detail)
