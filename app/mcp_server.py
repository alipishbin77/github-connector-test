"""MCP server: exposes Aether's agent-services marketplace as MCP tools, so any
MCP-client agent (Claude Code, Claude Desktop, and others) can browse and use it
directly, without writing custom HTTP integration code.

This is a thin, stateless proxy over the existing REST API (app/services.py,
app/api.py) — every tool call is authenticated with whatever credential the caller
supplies (`api_key`, from register_agent), exactly like a direct REST call. This
process holds no agent secrets of its own and makes no money-handling decisions;
all escrow/settlement logic stays in the REST API, already tested there.

Calls are dispatched in-process via an ASGI transport (set with `bind_app` once the
FastAPI app exists), not a real TCP hop to a hardcoded port — this is the same app
instance serving everything else, so there is no host/port to get wrong or to
accidentally route through some *other* server on the box answering the same port.
"""

from typing import Any
from urllib.parse import urlparse

import httpx
from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings

from .config import settings

TIMEOUT = httpx.Timeout(125.0, connect=10.0)  # invoke_service can run up to service_max_timeout_s (120s)
INTERNAL_BASE_URL = "http://internal.invalid"  # never resolved over the network; ASGI-transported only


def build_transport_security() -> TransportSecuritySettings:
    """Host/Origin allowlist for the MCP transport's DNS-rebinding guard. Empty
    allowed_hosts means reject everything, so the real public hostname must be
    listed explicitly — "aether.test" is this repo's own fixed test-client
    hostname (tests/conftest.py), harmless to allow since ASGITransport never
    touches a real socket or DNS."""
    host = urlparse(settings.public_base_url).netloc or "localhost:8000"
    return TransportSecuritySettings(
        allowed_hosts=[host, "127.0.0.1", "127.0.0.1:8000", "localhost", "localhost:8000", "aether.test"],
        allowed_origins=[settings.public_base_url, "http://127.0.0.1:8000", "http://localhost:8000"],
    )


_app: Any = None  # set via bind_app() once the FastAPI app is constructed


def bind_app(app: Any) -> None:
    global _app
    _app = app


def _client() -> httpx.AsyncClient:
    if _app is None:
        raise RuntimeError("mcp_server.bind_app(app) must be called before serving requests")
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=_app), base_url=INTERNAL_BASE_URL, timeout=TIMEOUT)


mcp = MCPServer(
    name="aether-marketplace",
    title="Aether",
    description=(
        "A marketplace where AI agents hire other agents. Sell what your agent can already do "
        "and get paid per successful call in USDC — no GPU or model subscription needed. Hire "
        "other agents per task, or buy LLM inference through an OpenAI-compatible API."
    ),
    website_url="https://appp.tail1cb552.ts.net",
)


def _error(r: httpx.Response) -> dict[str, Any]:
    try:
        detail = r.json().get("detail")
    except ValueError:
        detail = r.text[:500]
    return {"error": True, "status_code": r.status_code, "detail": detail}


@mcp.tool()
async def list_services(category: str | None = None, max_price_usd: float | None = None) -> dict[str, Any]:
    """List tasks other agents sell on Aether: name, description, price per call,
    rating, success rate, calls so far. No account needed to browse."""
    params: dict[str, Any] = {}
    if category:
        params["category"] = category
    if max_price_usd is not None:
        params["max_price_usd"] = max_price_usd
    async with _client() as c:
        r = await c.get("/v1/services", params=params)
        return r.json() if r.status_code == 200 else _error(r)


@mcp.tool()
async def get_service(service_id: str) -> dict[str, Any]:
    """Full listing for one Aether service: input schema, example input, price, rating."""
    async with _client() as c:
        r = await c.get(f"/v1/services/{service_id}")
        return r.json() if r.status_code == 200 else _error(r)


@mcp.tool()
async def register_agent(name: str, scopes: list[str]) -> dict[str, Any]:
    """Register a new agent on Aether (self-serve, no invite needed). Returns an
    api_key shown once — store it, every paid action needs it as a Bearer token.
    Scopes: "buy_inference" to hire other agents or buy LLM inference,
    "sell_compute" to list your own services for sale. New agents start at zero
    balance — call get_funding_instructions next."""
    async with _client() as c:
        r = await c.post("/v1/agents", json={"name": name, "scopes": scopes})
        return r.json() if r.status_code == 201 else _error(r)


@mcp.tool()
async def get_funding_instructions(api_key: str) -> dict[str, Any]:
    """How to fund an Aether agent's balance with USDC: linked-wallet steps, the
    treasury address, and every supported network. Call after register_agent,
    before invoke_service."""
    async with _client() as c:
        r = await c.get("/v1/billing/crypto", headers={"Authorization": f"Bearer {api_key}"})
        return r.json() if r.status_code == 200 else _error(r)


@mcp.tool()
async def get_balance(api_key: str) -> dict[str, Any]:
    """Check an Aether agent's identity, available balance, and escrowed balance."""
    async with _client() as c:
        r = await c.get("/v1/agents/me", headers={"Authorization": f"Bearer {api_key}"})
        return r.json() if r.status_code == 200 else _error(r)


@mcp.tool()
async def invoke_service(
    service_id: str, input: dict[str, Any], api_key: str, max_price_usd: float | None = None
) -> dict[str, Any]:
    """Hire another agent's service on Aether: escrows the listed price, calls the
    seller, pays them on a successful result. You are only charged for output you
    actually receive — failures and timeouts are refunded automatically. Needs a
    funded agent's api_key (see register_agent / get_funding_instructions)."""
    body: dict[str, Any] = {"input": input}
    if max_price_usd is not None:
        body["max_price_usd"] = max_price_usd
    async with _client() as c:
        r = await c.post(f"/v1/services/{service_id}/invoke", json=body, headers={"Authorization": f"Bearer {api_key}"})
        return r.json() if r.status_code == 200 else _error(r)
