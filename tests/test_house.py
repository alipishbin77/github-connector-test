"""House services: the four tasks the platform sells itself.

Each test calls the seller endpoint the way the clearinghouse does — compact
JSON body, single-use token bound to that body — because that is the only
contract these endpoints have.

Nothing here touches the live network. DNS is answered by `zone`, TLS by `tls`
and HTTP by `web`; name resolution itself goes through `hostmap`, which swaps
`socket.getaddrinfo` rather than `netguard`, so every test still runs the real
address rules. A previous PR broke because CI resolves differently from the
box this runs on.
"""

import hashlib
import ipaddress
import json
import socket
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID
from fastapi import HTTPException

from app import auth, netguard
from app import house as app_house
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

SLUGS = ("webpage-to-markdown", "json-schema-validate", "usdc-balance", "domain-trust-audit")
PRICE_USD = {"domain-trust-audit": "0.10"}  # the audit is the one house service that is not a fraction of a cent


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
    """One agent listing every house service, as the operator would. It must
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
                "price_usd": PRICE_USD.get(slug, "0.002"),
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
        await app_house._house_seller(created.json()["service_id"], "webpage-to-markdown")
    assert caught.value.status_code == 401
    assert "not owned by the house agent" in str(caught.value.detail)


# ------------------------------------------------------------ domain-trust-audit

AUDITED = "example.test"
AUDITED_IP = "93.184.216.34"


@pytest.fixture
def hostmap(monkeypatch):
    """Name resolution without a network. `socket.getaddrinfo` is what the event
    loop's resolver calls, so swapping it leaves every rule in `netguard` — the
    thing actually under test — running for real. Numeric hosts are passed
    through to the real resolver, which handles them locally."""
    mapping: dict[str, list[str]] = {}
    real = socket.getaddrinfo

    def fake(host, port, family=0, type=0, proto=0, flags=0):  # noqa: A002 - socket's own signature
        name = str(host).strip("[]")
        if name in mapping:
            return [
                (
                    socket.AF_INET6 if ":" in address else socket.AF_INET,
                    socket.SOCK_STREAM,
                    socket.IPPROTO_TCP,
                    "",
                    (address, port),
                )
                for address in mapping[name]
            ]
        try:
            ipaddress.ip_address(name)
        except ValueError:
            raise socket.gaierror(socket.EAI_NONAME, "Name or service not known") from None
        return real(host, port, family, type, proto, flags)

    monkeypatch.setattr(socket, "getaddrinfo", fake)
    return mapping


class FakeZone:
    """The DNS answers one domain has, keyed (qname, rdtype). Anything not
    listed is NXDOMAIN, which is what an unused DKIM selector really returns."""

    def __init__(self):
        self.records: dict[tuple[str, str], app_house.DnsAnswer] = {}
        self.asked: list[tuple[str, str]] = []

    def set(self, qname: str, rdtype: str, *values: str, status: str = "ok", detail: str = ""):
        self.records[(qname, rdtype)] = app_house.DnsAnswer(status, tuple(values), detail)

    async def lookup(self, qname: str, rdtype: str) -> app_house.DnsAnswer:
        self.asked.append((qname, rdtype))
        return self.records.get((qname, rdtype), app_house.DnsAnswer("nxdomain"))


@pytest.fixture
def zone(monkeypatch):
    fake = FakeZone()
    monkeypatch.setattr(app_house, "_dns_lookup", fake.lookup)
    return fake


class FakeTls:
    """Stands in for the handshake. Records the address it was asked to connect
    to, which is the point of most of the SSRF assertions below."""

    def __init__(self):
        self.calls: list[tuple[str, str, bool]] = []
        self.der: bytes | None = None
        self.version = "TLSv1.3"
        self.error: BaseException | None = None
        self.verify_error: BaseException | None = None

    async def handshake(self, address: str, server_hostname: str, *, verify: bool) -> tuple[bytes, str]:
        self.calls.append((address, server_hostname, verify))
        if self.error is not None:
            raise self.error
        if verify and self.verify_error is not None:
            raise self.verify_error
        assert self.der is not None, "the test did not set a certificate"
        return self.der, self.version


@pytest.fixture
def tls(monkeypatch):
    fake = FakeTls()
    fake.der = make_certificate([AUDITED])
    monkeypatch.setattr(app_house, "_tls_handshake", fake.handshake)
    return fake


def make_certificate(names: list[str], *, days_left: int = 90, age_days: int = 1) -> bytes:
    """A throwaway certificate in DER. EC, not RSA: key generation happens in
    every test that touches TLS and this box has no memory to spare."""
    key = ec.generate_private_key(ec.SECP256R1())
    now = datetime.now(UTC)
    return (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, names[0])]))
        .issuer_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Test Issuing CA")]))
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(days=age_days))
        .not_valid_after(now + timedelta(days=days_left))
        .add_extension(x509.SubjectAlternativeName([x509.DNSName(name) for name in names]), critical=False)
        .sign(key, hashes.SHA256())
        .public_bytes(serialization.Encoding.DER)
    )


SECURE_HEADERS = {
    "strict-transport-security": "max-age=63072000; includeSubDomains; preload",
    "content-security-policy": "default-src 'self'",
    "x-content-type-options": "nosniff",
    "x-frame-options": "DENY",
    "referrer-policy": "strict-origin-when-cross-origin",
}


def healthy_zone(zone: FakeZone, domain: str = AUDITED) -> None:
    zone.set(domain, "MX", "10 mail.example.test.", "20 mail2.example.test.")
    zone.set("mail.example.test", "A", AUDITED_IP)
    zone.set("mail2.example.test", "A", "93.184.216.35")
    zone.set(domain, "TXT", "v=spf1 include:_spf.example.test -all", "google-site-verification=abc")
    zone.set(domain, "DNSKEY", "257 3 13 mdsswUyr3DPW132mOi8V9xESWE8jTo0d")
    zone.set(f"_dmarc.{domain}", "TXT", "v=DMARC1; p=reject; rua=mailto:dmarc@example.test; pct=100")
    zone.set(f"default._domainkey.{domain}", "TXT", "v=DKIM1; k=rsa; p=MIIBIjANBgkq")


async def audit(client, house_fixture, payload, call_id=None):
    return await invoke(client, house_fixture, "domain-trust-audit", payload, call_id=call_id)


# ------------------------------------------------------- audit: the happy path


async def test_audit_returns_a_well_formed_verdict(client, house, web, hostmap, zone, tls):
    hostmap[AUDITED] = [AUDITED_IP]
    healthy_zone(zone)
    web.responses["/"] = lambda r: httpx.Response(200, headers=SECURE_HEADERS, text="hello")

    r = await audit(client, house, {"domain": AUDITED})

    assert r.status_code == 200, r.text
    out = r.json()["output"]
    assert out["domain"] == AUDITED
    assert (out["verdict"], out["score"]) == ("pass", 100)
    assert out["findings"] == []
    assert out["resolved_addresses"] == [AUDITED_IP]
    assert set(out["checks"]) == {"mx", "spf", "dmarc", "dkim", "dnssec", "tls", "headers"}
    assert out["checks"]["mx"]["records"][0] == {"preference": 10, "host": "mail.example.test"}
    assert out["checks"]["mx"]["resolvable"] == ["mail.example.test", "mail2.example.test"]
    assert out["checks"]["spf"]["policy"] == "fail" and out["checks"]["spf"]["permissive"] is False
    assert out["checks"]["dmarc"]["policy"] == "reject" and out["checks"]["dmarc"]["reporting"] is True
    assert out["checks"]["dkim"]["selectors_answered"] == ["default"]
    assert out["checks"]["dnssec"]["present"] is True
    assert out["checks"]["tls"]["hostname_match"] is True
    assert out["checks"]["tls"]["chain_trusted"] is True
    assert out["checks"]["tls"]["tls_version"] == "TLSv1.3"
    assert out["checks"]["tls"]["days_until_expiry"] in (88, 89, 90)
    assert "Test Issuing CA" in out["checks"]["tls"]["issuer"]
    assert out["checks"]["headers"]["missing"] == []
    assert out["checks"]["headers"]["headers"]["x-frame-options"]["value"] == "DENY"
    assert out["dns_queries_used"] <= house_module_budget()


def house_module_budget() -> int:
    return app_house.MAX_DNS_QUERIES


async def test_audit_accepts_a_url_and_takes_its_hostname(client, house, web, hostmap, zone, tls):
    hostmap[AUDITED] = [AUDITED_IP]
    healthy_zone(zone)
    web.responses["/"] = lambda r: httpx.Response(200, headers=SECURE_HEADERS, text="hi")

    r = await audit(client, house, {"domain": "https://Example.TEST:443/some/path?q=1"})

    assert r.status_code == 200, r.text
    assert r.json()["output"]["domain"] == AUDITED


async def test_audit_flags_a_permissive_spf_and_a_missing_dmarc(client, house, web, hostmap, zone, tls):
    hostmap[AUDITED] = [AUDITED_IP]
    zone.set(AUDITED, "MX", "10 mail.example.test.")
    zone.set("mail.example.test", "A", AUDITED_IP)
    zone.set(AUDITED, "TXT", "v=spf1 +all")
    web.responses["/"] = lambda r: httpx.Response(200, headers={"x-frame-options": "DENY"}, text="hi")

    r = await audit(client, house, {"domain": AUDITED})

    assert r.status_code == 200, r.text
    out = r.json()["output"]
    assert out["verdict"] == "fail"
    assert out["score"] < 60
    assert out["checks"]["spf"]["all_qualifier"] == "+" and out["checks"]["spf"]["permissive"] is True
    assert out["checks"]["dmarc"]["status"] == "absent"
    by_check = {(f["check"], f["severity"]) for f in out["findings"]}
    assert ("spf", "high") in by_check
    assert ("dmarc", "high") in by_check
    assert ("headers", "medium") in by_check  # HSTS


async def test_audit_reports_two_spf_records_as_a_permanent_error(client, house, web, hostmap, zone, tls):
    hostmap[AUDITED] = [AUDITED_IP]
    healthy_zone(zone)
    zone.set(AUDITED, "TXT", "v=spf1 -all", "v=spf1 include:other.test ~all")
    web.responses["/"] = lambda r: httpx.Response(200, headers=SECURE_HEADERS, text="hi")

    out = (await audit(client, house, {"domain": AUDITED})).json()["output"]

    assert out["checks"]["spf"]["status"] == "multiple"
    assert out["verdict"] == "fail"


async def test_audit_says_dkim_absence_is_not_proof_of_absence(client, house, web, hostmap, zone, tls):
    hostmap[AUDITED] = [AUDITED_IP]
    healthy_zone(zone)
    del zone.records[(f"default._domainkey.{AUDITED}", "TXT")]
    web.responses["/"] = lambda r: httpx.Response(200, headers=SECURE_HEADERS, text="hi")

    out = (await audit(client, house, {"domain": AUDITED})).json()["output"]

    dkim = out["checks"]["dkim"]
    assert dkim["status"] == "not_found"
    assert dkim["selectors_answered"] == []
    assert dkim["selectors_probed"] == list(app_house.DKIM_SELECTORS)
    assert "not proof" in dkim["note"]
    # A low finding only. The verdict still passes — an unanswered selector is not
    # evidence of anything — but the score records that we looked and found nothing.
    assert out["verdict"] == "pass"
    assert out["score"] < 100
    assert {(f["check"], f["severity"]) for f in out["findings"]} == {("dkim", "low")}


# --------------------------------------------- audit: failures that are results


async def test_audit_reports_no_mx_and_no_tls_listener_rather_than_crashing(client, house, web, hostmap, zone, tls):
    hostmap[AUDITED] = [AUDITED_IP]
    zone.set(AUDITED, "MX", status="empty")  # the name exists; it just publishes no MX
    zone.set(AUDITED, "TXT", "v=spf1 -all")
    tls.error = ConnectionRefusedError("connection refused")

    def refused(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    web.responses["/"] = refused

    r = await audit(client, house, {"domain": AUDITED})

    assert r.status_code == 200, r.text
    out = r.json()["output"]
    assert out["checks"]["mx"]["status"] == "absent"
    assert out["checks"]["tls"]["status"] == "unreachable"
    assert "ConnectionRefusedError" in out["checks"]["tls"]["error"]
    assert out["checks"]["headers"]["status"] == "unreachable"
    assert out["verdict"] == "fail"  # no MX, no DMARC — a result, not a 5xx


async def test_audit_reports_nxdomain_as_a_result(client, house, web, hostmap, zone, tls):
    r = await audit(client, house, {"domain": "no-such-domain.test"})

    assert r.status_code == 200, r.text
    out = r.json()["output"]
    assert out["resolved_addresses"] == []
    assert out["checks"]["tls"]["status"] == "unreachable"
    assert out["checks"]["headers"]["status"] == "unreachable"
    assert out["checks"]["mx"]["status"] == "nxdomain"
    assert out["verdict"] == "fail"
    assert tls.calls == [] and web.requested == []


async def test_audit_reports_a_null_mx_as_deliberate(client, house, web, hostmap, zone, tls):
    hostmap[AUDITED] = [AUDITED_IP]
    healthy_zone(zone)
    zone.set(AUDITED, "MX", "0 .")
    web.responses["/"] = lambda r: httpx.Response(200, headers=SECURE_HEADERS, text="hi")

    out = (await audit(client, house, {"domain": AUDITED})).json()["output"]

    assert out["checks"]["mx"]["status"] == "null_mx"
    # RFC 7505 is a statement, not a fault, so it costs a low finding and no more.
    assert out["verdict"] == "pass"
    assert {(f["check"], f["severity"]) for f in out["findings"]} == {("mx", "low")}


async def test_audit_reports_unresolvable_mail_hosts(client, house, web, hostmap, zone, tls):
    hostmap[AUDITED] = [AUDITED_IP]
    healthy_zone(zone)
    del zone.records[("mail.example.test", "A")]
    del zone.records[("mail2.example.test", "A")]
    web.responses["/"] = lambda r: httpx.Response(200, headers=SECURE_HEADERS, text="hi")

    out = (await audit(client, house, {"domain": AUDITED})).json()["output"]

    assert out["checks"]["mx"]["resolvable"] == []
    assert ("mx", "high") in {(f["check"], f["severity"]) for f in out["findings"]}


@pytest.mark.parametrize(
    ("days_left", "expected"),
    [(5, "the certificate expires in"), (-2, "the certificate expired on")],
)
async def test_audit_flags_certificate_expiry(client, house, web, hostmap, zone, tls, days_left, expected):
    hostmap[AUDITED] = [AUDITED_IP]
    healthy_zone(zone)
    tls.der = make_certificate([AUDITED], days_left=days_left, age_days=400)
    web.responses["/"] = lambda r: httpx.Response(200, headers=SECURE_HEADERS, text="hi")

    out = (await audit(client, house, {"domain": AUDITED}, call_id=f"expiry-{days_left}")).json()["output"]

    assert out["checks"]["tls"]["expired"] is (days_left < 0)
    assert any(expected in finding["message"] for finding in out["findings"]), out["findings"]


async def test_audit_flags_a_certificate_for_the_wrong_hostname(client, house, web, hostmap, zone, tls):
    hostmap[AUDITED] = [AUDITED_IP]
    healthy_zone(zone)
    tls.der = make_certificate(["somewhere-else.test"])
    web.responses["/"] = lambda r: httpx.Response(200, headers=SECURE_HEADERS, text="hi")

    out = (await audit(client, house, {"domain": AUDITED})).json()["output"]

    assert out["checks"]["tls"]["hostname_match"] is False
    assert out["verdict"] == "fail"


async def test_audit_reports_an_untrusted_chain_with_the_certificate_details(client, house, web, hostmap, zone, tls):
    """A chain that does not verify is the finding, so the certificate behind it
    is still read — over a second, deliberately unverified handshake."""
    hostmap[AUDITED] = [AUDITED_IP]
    healthy_zone(zone)
    tls.verify_error = ssl_verification_error("self-signed certificate")
    web.responses["/"] = lambda r: httpx.Response(200, headers=SECURE_HEADERS, text="hi")

    out = (await audit(client, house, {"domain": AUDITED})).json()["output"]

    assert out["checks"]["tls"]["chain_trusted"] is False
    assert "self-signed" in out["checks"]["tls"]["chain_error"]
    assert out["checks"]["tls"]["hostname_match"] is True  # details still read
    assert [verify for _, _, verify in tls.calls] == [True, False]
    assert {address for address, _, _ in tls.calls} == {AUDITED_IP}  # the retry is not a second lookup
    assert out["verdict"] == "fail"


def ssl_verification_error(message: str):
    import ssl

    error = ssl.SSLCertVerificationError(message)
    error.verify_message = message
    return error


async def test_audit_handles_a_wildcard_certificate(client, house, web, hostmap, zone, tls):
    hostmap["mail.example.test"] = [AUDITED_IP]
    zone.set("mail.example.test", "TXT", "v=spf1 -all")
    zone.set("_dmarc.mail.example.test", "TXT", "v=DMARC1; p=reject; rua=mailto:a@b.test")
    tls.der = make_certificate(["*.example.test"])
    web.responses["/"] = lambda r: httpx.Response(200, headers=SECURE_HEADERS, text="hi")

    out = (await audit(client, house, {"domain": "mail.example.test"})).json()["output"]

    assert out["checks"]["tls"]["hostname_match"] is True


# ------------------------------------------------------------------ audit: SSRF


@pytest.mark.parametrize("host", PRIVATE_HOSTS)
async def test_audit_refuses_a_private_target_before_connecting(client, house, web, hostmap, zone, tls, host):
    """Both connections — TLS and the header fetch — are gated on one netguard
    resolution, so a non-public target never reaches either."""
    r = await audit(client, house, {"domain": host}, call_id=f"audit-ssrf-{host}")

    assert r.status_code == 422, r.text
    assert r.json()["detail"]["error"] == "refused_domain"
    assert tls.calls == []  # no TLS socket
    assert web.requested == []  # no HTTP request
    assert zone.asked == []  # not even a DNS question was paid for


@pytest.mark.parametrize("address", ["127.0.0.1", "169.254.169.254", "10.77.0.1", "100.114.15.67", "::1"])
async def test_audit_refuses_a_name_that_resolves_somewhere_private(client, house, web, hostmap, zone, tls, address):
    """The realistic attack: a perfectly ordinary-looking domain whose A record
    the attacker controls. 100.64.0.0/10 is this host's own Tailscale range and
    is neither is_private nor is_global, which is why netguard demands is_global."""
    hostmap["evil.test"] = [address]

    r = await audit(client, house, {"domain": "evil.test"}, call_id=f"audit-rebind-{address}")

    assert r.status_code == 422, r.text
    assert r.json()["detail"]["error"] == "refused_domain"
    assert str(ipaddress.ip_address(address)) in r.json()["detail"]["error_description"]
    assert tls.calls == [] and web.requested == []


async def test_audit_refuses_a_name_with_one_private_answer_among_public_ones(client, house, web, hostmap, zone, tls):
    """One private record is all an attacker needs, so a mixed answer is refused
    outright rather than filtered down to the public addresses."""
    hostmap["mixed.test"] = [AUDITED_IP, "127.0.0.1"]

    r = await audit(client, house, {"domain": "mixed.test"})

    assert r.status_code == 422
    assert tls.calls == [] and web.requested == []


async def test_audit_tls_connects_only_to_the_address_netguard_approved(client, house, web, hostmap, zone, tls):
    """The hostname is resolved once. What TLS connects to is that result, not a
    second lookup that could answer differently — the hostname travels only as
    SNI and as the name the certificate is checked against."""
    hostmap[AUDITED] = [AUDITED_IP]
    healthy_zone(zone)
    web.responses["/"] = lambda r: httpx.Response(200, headers=SECURE_HEADERS, text="hi")

    r = await audit(client, house, {"domain": AUDITED})

    assert r.status_code == 200, r.text
    assert tls.calls == [(AUDITED_IP, AUDITED, True)]


async def test_audit_header_redirect_to_a_private_address_is_refused(client, house, web, hostmap, zone, tls):
    """A public site that 302s at the metadata service. The hop is re-checked,
    the audit still answers, and the header check reports the refusal."""
    hostmap[AUDITED] = [AUDITED_IP]
    healthy_zone(zone)
    web.responses["/"] = lambda r: httpx.Response(301, headers={"location": "https://169.254.169.254/latest/meta-data/"})

    r = await audit(client, house, {"domain": AUDITED})

    assert r.status_code == 200, r.text
    headers = r.json()["output"]["checks"]["headers"]
    assert headers["status"] == "refused"
    assert "169.254.169.254" in headers["error"]
    assert web.requested == [f"https://{AUDITED}/"]  # the metadata service was never contacted


async def test_audit_header_redirect_down_to_plain_http_is_refused(client, house, web, hostmap, zone, tls):
    """This check is about the headers a visitor gets over https. A hop that
    drops to http is not that, and is refused rather than followed."""
    hostmap[AUDITED] = [AUDITED_IP]
    healthy_zone(zone)
    web.responses["/"] = lambda r: httpx.Response(301, headers={"location": f"http://{AUDITED}/"})

    out = (await audit(client, house, {"domain": AUDITED})).json()["output"]

    assert out["checks"]["headers"]["status"] == "refused"
    assert "https" in out["checks"]["headers"]["error"]


async def test_audit_header_redirects_are_bounded(client, house, web, hostmap, zone, tls):
    hostmap[AUDITED] = [AUDITED_IP]
    healthy_zone(zone)
    web.responses["/"] = lambda r: httpx.Response(302, headers={"location": f"https://{AUDITED}/"})

    out = (await audit(client, house, {"domain": AUDITED})).json()["output"]

    assert out["checks"]["headers"]["status"] == "too_many_redirects"
    assert len(web.requested) == app_house.MAX_HEADER_REDIRECTS + 1


# -------------------------------------------------- audit: limits and bad input


async def test_audit_dns_budget_is_bounded(client, house, web, hostmap, zone, tls):
    """A domain with many mail hosts must not let one $0.10 call ask DNS
    forever."""
    hostmap[AUDITED] = [AUDITED_IP]
    healthy_zone(zone)
    zone.set(AUDITED, "MX", *[f"{i * 10} mx{i}.example.test." for i in range(1, 20)])
    web.responses["/"] = lambda r: httpx.Response(200, headers=SECURE_HEADERS, text="hi")

    out = (await audit(client, house, {"domain": AUDITED})).json()["output"]

    assert len(zone.asked) <= app_house.MAX_DNS_QUERIES
    assert out["dns_queries_used"] <= app_house.MAX_DNS_QUERIES
    assert len(out["checks"]["mx"]["probed"]) == app_house.MAX_MX_RESOLVED  # only the lowest preferences


async def test_audit_timeouts_are_reported_without_a_5xx(client, house, web, hostmap, zone, tls):
    hostmap[AUDITED] = [AUDITED_IP]
    for qname, rdtype in [(AUDITED, "MX"), (AUDITED, "TXT"), (f"_dmarc.{AUDITED}", "TXT"), (AUDITED, "DNSKEY")]:
        zone.set(qname, rdtype, status="error", detail="the nameserver did not answer in time")
    tls.error = TimeoutError()

    def slow(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow", request=request)

    web.responses["/"] = slow

    r = await audit(client, house, {"domain": AUDITED})

    assert r.status_code == 200, r.text
    out = r.json()["output"]
    assert out["checks"]["mx"]["status"] == "error"
    assert out["checks"]["spf"]["status"] == "error"
    assert out["checks"]["dmarc"]["status"] == "error"
    assert out["checks"]["tls"]["status"] == "unreachable"
    assert "no TLS handshake" in out["checks"]["tls"]["error"]
    assert out["checks"]["headers"]["status"] == "timeout"


async def test_audit_survives_a_peer_that_sends_rubbish_instead_of_a_certificate(client, house, web, hostmap, zone, tls):
    hostmap[AUDITED] = [AUDITED_IP]
    healthy_zone(zone)
    tls.der = b"not a certificate"
    web.responses["/"] = lambda r: httpx.Response(200, headers=SECURE_HEADERS, text="hi")

    r = await audit(client, house, {"domain": AUDITED})

    assert r.status_code == 200, r.text
    assert r.json()["output"]["checks"]["tls"]["status"] == "unreadable_certificate"


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"domain": 42},
        {"domain": ""},
        {"domain": "   "},
        {"domain": "not a domain"},
        {"domain": "localhost"},  # no dot: nothing to audit, and netguard would refuse it anyway
        {"domain": "http://"},
        {"domain": "-leading-hyphen.test"},
        {"domain": "a" * 70 + ".test"},  # a label may not exceed 63 bytes
        {"domain": "under_score.test"},
        None,
        "just a string",
    ],
)
async def test_audit_malformed_input_is_a_clean_4xx(client, house, web, hostmap, zone, tls, payload):
    r = await audit(client, house, payload, call_id=f"audit-bad-{payload}")

    assert r.status_code == 422, r.text
    detail = r.json()["detail"]
    assert set(detail) == {"error", "error_description"}
    assert detail["error"] in ("invalid_input", "invalid_domain")
    assert "Traceback" not in detail["error_description"]
    assert web.requested == [] and tls.calls == [] and zone.asked == []


def test_normalise_domain_accepts_bare_names_and_urls():
    for value, expected in [
        ("Example.COM", "example.com"),
        ("example.com.", "example.com"),
        ("  example.com  ", "example.com"),
        ("https://example.com/a/b?c=d#e", "example.com"),
        ("http://example.com:8080/", "example.com"),
        ("example.com/some/path", "example.com"),
        ("example.com:8443", "example.com"),
        ("xn--bcher-kva.example", "xn--bcher-kva.example"),
        ("bücher.example", "xn--bcher-kva.example"),
        ("127.0.0.1", "127.0.0.1"),  # handed to netguard, which is what refuses it
    ]:
        assert app_house._normalise_domain(value) == expected, value


def test_hostname_matching_follows_one_wildcard_label():
    assert app_house._hostname_matches("a.example.com", ["*.example.com"])
    assert app_house._hostname_matches("example.com", ["example.com."])
    assert not app_house._hostname_matches("b.a.example.com", ["*.example.com"])
    assert not app_house._hostname_matches("example.com", ["*.example.com"])
    assert not app_house._hostname_matches("example.com.evil.test", ["example.com"])
