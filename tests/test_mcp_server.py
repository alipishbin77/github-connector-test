"""The MCP server (app/mcp_server.py) is a thin proxy over the REST API already
covered elsewhere — these tests only check the MCP wiring itself: tools are
discoverable, and a read-only + a stateful call both round-trip correctly through
a real MCP client session, in-process (no socket, no hardcoded port)."""

import json

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from .helpers import new_agent


async def _session(app):
    http_client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://aether.test")
    return streamable_http_client("http://aether.test/mcp", http_client=http_client)


async def test_lists_every_tool(app):
    async with await _session(app) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        tools = {t.name for t in (await session.list_tools()).tools}
        assert tools == {
            "list_services",
            "get_service",
            "register_agent",
            "get_funding_instructions",
            "get_balance",
            "invoke_service",
        }


async def test_get_service_reflects_the_real_catalogue(app, client):
    # Uses get_service (fetch by id) rather than list_services: the shared
    # session-scoped DB accumulates services from every other test in the
    # suite, and list_services' default page size isn't guaranteed to include
    # one created this late — not something this MCP wiring test should care about.
    seller = await new_agent(client, "mcp-test-seller", ["sell_compute"])
    listed = (
        (
            await client.post(
                "/v1/services",
                headers=seller["headers"],
                json={
                    "name": "MCP test service",
                    "description": "A service listed purely to verify the MCP tool sees it.",
                    "category": "other",
                    "price_usd": "0.01",
                    "endpoint_url": "http://aether.test/v1/house/webpage-to-markdown",
                },
            )
        )
        .raise_for_status()
        .json()
    )

    async with await _session(app) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        result = await session.call_tool("get_service", {"service_id": listed["service_id"]})
        data = json.loads(result.content[0].text)
        assert data["service_id"] == listed["service_id"]
        assert data["name"] == "MCP test service"


async def test_register_agent_then_get_balance_round_trips(app):
    async with await _session(app) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        reg = json.loads(
            (await session.call_tool("register_agent", {"name": "mcp-test-agent", "scopes": ["buy_inference"]})).content[0].text
        )
        assert reg["scopes"] == ["buy_inference"]

        bal = json.loads((await session.call_tool("get_balance", {"api_key": reg["api_key"]})).content[0].text)
        assert bal["agent_id"] == reg["agent_id"]
        assert bal["available_usd"] == "$0.000000000"


async def test_get_balance_rejects_a_bad_api_key(app):
    async with await _session(app) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        result = await session.call_tool("get_balance", {"api_key": "not_a_real.key"})
        data = json.loads(result.content[0].text)
        assert data["error"] is True
        assert data["status_code"] == 401
