"""House services: three tasks the platform itself sells, so the catalogue is
never empty. No GPU, no LLM, no paid API.

  POST /v1/house/webpage-to-markdown  {"url": "https://..."}
       -> the readable content of that page as Markdown text
  POST /v1/house/json-schema-validate {"schema": {...}, "data": ...}
       -> {"valid": bool, "errors": [...]}
  POST /v1/house/usdc-balance         {"address": "0x...", "network": "base"}
       -> that address's native-USDC balance on one EVM network

These are ordinary *seller* endpoints: the clearinghouse calls them exactly as
it calls a third party's, with the compact JSON body
`{"call_id", "service_id", "input"}` and a single-use bearer token, and they
answer `{"output": ...}`. Anything non-2xx refunds the buyer, so every
rejection below costs the buyer nothing.

Token verification follows `seller_gateway.verify_delivery_token`: RS256,
audience/issuer, single use by `jti`, and bound to the exact request body by
its sha256. It differs in one deliberate way — the platform *is* the issuer,
so the token is checked against the same signing key that
`/.well-known/jwks.json` publishes instead of fetching that document from
ourselves over HTTP. The audience is not hard-coded: it is derived from the
seller who listed the `service_id` in the body, and that service must be
registered against this very endpoint, so a token minted for some other
seller's service cannot be spent here.

`webpage-to-markdown` makes the server fetch an address a stranger chose, so
every URL — and every redirect hop — goes through `netguard`, which resolves
first and refuses anything that is not a public address. Responses are size
capped and time bounded.
"""

import hashlib
import hmac
import json
import logging
import time
from decimal import Decimal
from typing import Any, TypeVar
from urllib.parse import urljoin, urlparse

import httpx
import jwt
from fastapi import APIRouter, HTTPException, Request
from jsonschema import Draft202012Validator, SchemaError
from jsonschema.validators import validator_for
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from . import auth, netguard
from .config import settings
from .crypto_payments import ADDRESS_RE, PRESETS, EvmClient, Network
from .db import SessionLocal
from .models import Service
from .readable import html_to_markdown

log = logging.getLogger("aether.house")

HOUSE_PREFIX = "/v1/house"
router = APIRouter(prefix=HOUSE_PREFIX, tags=["house services"])

MAX_BODY_BYTES = 256 * 1024  # the platform only ever sends {call_id, service_id, input}
MAX_PAGE_BYTES = 3_000_000
MAX_REDIRECTS = 3
FETCH_TIMEOUT = httpx.Timeout(10.0, connect=5.0)
USER_AGENT = "Aether-house-services/1.0 (agent marketplace; webpage-to-markdown)"
READABLE_TYPES = ("text/html", "application/xhtml+xml", "text/plain", "text/markdown", "text/xml", "application/xml")
MAX_SCHEMA_ERRORS = 50

ModelT = TypeVar("ModelT", bound=BaseModel)


# ----------------------------------------------------------------- input shapes


class PageIn(BaseModel):
    model_config = ConfigDict(extra="ignore")
    url: str = Field(min_length=8, max_length=2048)


class SchemaIn(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)
    json_schema: dict[str, Any] | bool = Field(alias="schema")
    data: Any = None


class BalanceIn(BaseModel):
    model_config = ConfigDict(extra="ignore")
    address: str = Field(min_length=42, max_length=42)
    network: str | None = None


def _error(status: int, code: str, description: str) -> HTTPException:
    return HTTPException(status, detail={"error": code, "error_description": description})


def _parse(model: type[ModelT], value: Any) -> ModelT:
    """Validate the buyer's `input` into `model`, reporting the first problem
    in one line: a stack trace would be leaked to a stranger."""
    try:
        return model.model_validate(value)
    except ValidationError as exc:
        first = exc.errors()[0]
        field = ".".join(str(p) for p in first["loc"]) or "input"
        raise _error(422, "invalid_input", f"{field}: {first['msg']}") from None


# ------------------------------------------------------------------------ auth


def _unauthorized(description: str) -> HTTPException:
    return HTTPException(
        401,
        detail={"error": "invalid_token", "error_description": description},
        headers={"WWW-Authenticate": 'Bearer realm="aether-house"'},
    )


async def _house_seller(service_id: str, slug: str) -> str:
    """The agent that listed `service_id`, provided that listing points at this
    endpoint. Binds a token to one house service instead of all of them."""
    async with SessionLocal() as session:
        service = await session.get(Service, service_id)
    if service is None:
        raise _unauthorized("unknown service_id")
    if urlparse(service.endpoint_url).path.rstrip("/") != f"{router.prefix}/{slug}":
        raise _unauthorized(f"service {service_id} is not listed against {router.prefix}/{slug}")
    return service.seller_id


def _verify_token(token: str, raw_body: bytes, *, seller_id: str) -> dict:
    try:
        claims = jwt.decode(
            token,
            auth.signing_key().public_key,
            algorithms=[auth.ALGORITHM],
            audience=auth.delivery_audience(seller_id),
            issuer=settings.jwt_issuer,
            leeway=5,
            options={"require": ["exp", "iat", "nbf", "iss", "aud", "sub", "jti"]},
        )
    except jwt.InvalidTokenError as exc:
        raise _unauthorized(f"token rejected: {exc}") from None
    if claims.get("typ") != "service_call":
        raise _unauthorized("not a service-call token")
    if not hmac.compare_digest(str(claims.get("body_sha256", "")), hashlib.sha256(raw_body).hexdigest()):
        raise _unauthorized("request body does not match the token")
    return claims


async def _claim_once(request: Request, claims: dict) -> None:
    """Burn the token's jti. Redis, so it holds across workers and restarts."""
    ttl = max(1, int(claims["exp"]) - int(time.time()) + 5)
    if not await request.app.state.redis.set(f"{settings.redis_prefix}:house:jti:{claims['jti']}", "1", ex=ttl, nx=True):
        raise _unauthorized("token already used")


async def authorized_input(request: Request, slug: str) -> Any:
    """Verify the call token, then hand back the buyer's `input`. Nothing in
    this module does any work before this returns."""
    header = request.headers.get("authorization", "")
    if not header.lower().startswith("bearer "):
        raise _unauthorized("missing service-call token")
    raw = await request.body()
    if len(raw) > MAX_BODY_BYTES:
        raise _error(413, "body_too_large", f"the request body must be under {MAX_BODY_BYTES} bytes")
    try:
        body = json.loads(raw)
    except ValueError:
        raise _error(400, "invalid_request", "body is not JSON") from None
    if not isinstance(body, dict) or not isinstance(body.get("service_id"), str) or not isinstance(body.get("call_id"), str):
        raise _error(400, "invalid_request", 'body must be {"call_id", "service_id", "input"}')
    seller_id = await _house_seller(body["service_id"], slug)
    claims = _verify_token(header[7:].strip(), raw, seller_id=seller_id)
    if claims["sub"] != body["call_id"] or claims.get("service_id") != body["service_id"]:
        raise _unauthorized("token does not match the call it was sent with")
    await _claim_once(request, claims)
    log.info("HOUSE %s call %s", slug, body["call_id"])
    return body.get("input")


# ------------------------------------------------------------ webpage fetching


async def _fetch_page(http: httpx.AsyncClient, url: str) -> tuple[str, str, str]:
    """Fetch a buyer-supplied page. Returns (final_url, content_type, text).

    Redirects are followed by hand so that *every* hop is resolved and
    classified before it is connected to; a URL that is public now can still
    302 to 169.254.169.254."""
    for _ in range(MAX_REDIRECTS + 1):
        try:
            await netguard.check_public_url(url, require_https=False)
        except netguard.UnsafeURL as exc:
            raise _error(422, "refused_url", f"url {exc}") from None
        headers = {"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml,text/plain;q=0.8"}
        try:
            async with http.stream("GET", url, headers=headers, timeout=FETCH_TIMEOUT, follow_redirects=False) as resp:
                if resp.status_code in (301, 302, 303, 307, 308):
                    location = resp.headers.get("location")
                    if not location:
                        raise _error(502, "fetch_failed", f"the page answered HTTP {resp.status_code} without a location")
                    url = urljoin(str(resp.url), location)
                    continue
                if not 200 <= resp.status_code < 300:
                    raise _error(502, "fetch_failed", f"the page answered HTTP {resp.status_code}")
                content_type = resp.headers.get("content-type", "")
                if content_type and not content_type.lower().startswith(READABLE_TYPES):
                    raise _error(415, "not_a_page", f"cannot read {content_type.split(';')[0]}; this service reads web pages")
                declared = resp.headers.get("content-length")
                if declared and declared.isdigit() and int(declared) > MAX_PAGE_BYTES:
                    raise _error(413, "page_too_large", f"the page is larger than {MAX_PAGE_BYTES} bytes")
                body = bytearray()
                async for chunk in resp.aiter_bytes():
                    body += chunk
                    if len(body) > MAX_PAGE_BYTES:
                        raise _error(413, "page_too_large", f"the page is larger than {MAX_PAGE_BYTES} bytes")
                try:
                    text = bytes(body).decode(resp.charset_encoding or "utf-8", errors="replace")
                except LookupError:  # the page declared a charset Python does not know
                    text = bytes(body).decode("utf-8", errors="replace")
                return str(resp.url), content_type, text
        except httpx.TimeoutException:
            raise _error(504, "fetch_timeout", "the page did not answer in time") from None
        except httpx.HTTPError as exc:
            raise _error(502, "fetch_failed", f"could not fetch the page: {type(exc).__name__}") from None
    raise _error(502, "too_many_redirects", f"the page redirected more than {MAX_REDIRECTS} times")


# -------------------------------------------------------------------- services


@router.post("/webpage-to-markdown")
async def webpage_to_markdown(request: Request):
    """The readable content of a public web page, as Markdown."""
    body = _parse(PageIn, await authorized_input(request, "webpage-to-markdown"))
    final_url, content_type, text = await _fetch_page(request.app.state.http, body.url)
    if content_type.lower().startswith(("text/plain", "text/markdown")):
        return {"output": text.strip()}
    return {"output": html_to_markdown(text, base_url=final_url)}


@router.post("/json-schema-validate")
async def json_schema_validate(request: Request):
    """Validate `data` against a JSON Schema (draft 2020-12 by default; a
    `$schema` in the document picks its own draft)."""
    body = _parse(SchemaIn, await authorized_input(request, "json-schema-validate"))
    schema = body.json_schema
    validator_class = validator_for(schema, default=Draft202012Validator) if isinstance(schema, dict) else Draft202012Validator
    try:
        validator_class.check_schema(schema)
    except SchemaError as exc:
        raise _error(422, "invalid_schema", f"schema is not valid JSON Schema: {exc.message}") from None
    errors = sorted(validator_class(schema).iter_errors(body.data), key=lambda e: list(e.absolute_path))
    return {
        "output": {
            "valid": not errors,
            "errors": [
                {
                    "path": "/" + "/".join(str(p) for p in error.absolute_path),
                    "message": error.message,
                    "validator": str(error.validator),
                }
                for error in errors[:MAX_SCHEMA_ERRORS]
            ],
            "error_count": len(errors),
            "draft": validator_class.__name__,
        }
    }


def _network(key: str | None) -> Network:
    if key is None:
        return PRESETS["base"]
    network = PRESETS.get(key.strip().lower())
    if network is None:
        raise _error(422, "unknown_network", f"network must be one of: {', '.join(PRESETS)}")
    return network


@router.post("/usdc-balance")
async def usdc_balance(request: Request):
    """Native-USDC balance of an address on one EVM network, read straight
    from the token contract with `balanceOf`."""
    body = _parse(BalanceIn, await authorized_input(request, "usdc-balance"))
    if not ADDRESS_RE.match(body.address):
        raise _error(422, "invalid_address", "address must be a 0x-prefixed 20-byte hex address")
    network = _network(body.network)
    # Same JSON-RPC client the deposit rails use, over the app's shared pool.
    client = EvmClient(request.app.state.http, network.rpc_url)
    try:
        units = await client.token_balance(network.token_address, body.address)
    except Exception as exc:  # ChainError, and anything else the RPC throws at us
        log.warning("house usdc-balance %s: %s", network.key, exc)
        raise _error(502, "rpc_failed", f"could not read the balance on {network.name}") from None
    return {
        "output": {
            "address": body.address.lower(),
            "network": network.key,
            "chain_id": network.chain_id,
            "token": network.token_symbol,
            "token_contract": network.token_address,
            "decimals": network.decimals,
            "balance": str(Decimal(units) / 10**network.decimals),
            "balance_units": units,
        }
    }
