"""House services: four tasks the platform itself sells, so the catalogue is
never empty. No GPU, no LLM, no paid API.

  POST /v1/house/webpage-to-markdown  {"url": "https://..."}
       -> the readable content of that page as Markdown text
  POST /v1/house/json-schema-validate {"schema": {...}, "data": ...}
       -> {"valid": bool, "errors": [...]}
  POST /v1/house/usdc-balance         {"address": "0x...", "network": "base"}
       -> that address's native-USDC balance on one EVM network
  POST /v1/house/domain-trust-audit   {"domain": "example.com"}
       -> mail-authentication, TLS and security-header posture for one domain

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

`domain-trust-audit` connects to a stranger's host twice, over TLS and over
HTTPS, so it is the same risk again and obeys the same rule. It resolves the
domain **once**, through `netguard.resolve_public`, and the TLS socket is then
opened against one of the addresses that call already approved — the hostname
travels on as SNI only, so a second lookup cannot answer differently. A domain
that resolves anywhere non-public is refused outright; one that does not
resolve at all is a finding about that domain, not a refusal, which is why
`netguard` tells the two apart. The header fetch goes through the shared httpx
client (no new pool), pre-checked hop by hop like the page fetcher.

Every check degrades to a reported failure. No MX, no TLS listener and
NXDOMAIN are all *results*; only a malformed domain is a 4xx.
"""

import asyncio
import contextlib
import hashlib
import hmac
import ipaddress
import json
import logging
import re
import ssl
import time
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, NamedTuple, TypeVar
from urllib.parse import urljoin, urlparse

import dns.asyncresolver
import dns.exception
import dns.resolver
import httpx
import jwt
from cryptography import x509
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

# ---- domain-trust-audit. Every number here is a ceiling, not a target: the box
# has 3 GB and no swap, so one buyer must not be able to hold a worker open.
AUDIT_BUDGET_S = 20.0  # whole call, across every check
DNS_TIMEOUT_S = 3.0
MAX_DNS_QUERIES = 20  # 10 asked at once, then at most 6 MX address lookups
MAX_MX_RESOLVED = 3  # only the lowest-preference hosts are probed
TLS_TIMEOUT_S = 5.0
TLS_PORT = 443
HEADER_TIMEOUT = httpx.Timeout(5.0, connect=3.0)
MAX_HEADER_REDIRECTS = 2
MAX_HEADER_BYTES = 64 * 1024  # headers are all we keep; the body is drained and dropped
CERT_EXPIRY_WARN_DAYS = 30
MAX_TLS_ADDRESSES = 2  # a domain with a dead first A record still gets audited
MAX_REPORTED = 16  # a domain may publish 200 MX records; the buyer is not paying to receive them
CONNECT_PHASE_S = 10.0  # hard stop on TLS + headers together, whatever their own timeouts do
# DKIM keys live at a selector the sender chooses, so there is nothing to enumerate.
# Probing the common ones can only ever prove presence.
DKIM_SELECTORS = ("default", "google", "selector1", "selector2", "k1", "mail")
SECURITY_HEADERS = (
    "strict-transport-security",
    "content-security-policy",
    "x-content-type-options",
    "x-frame-options",
    "referrer-policy",
)
# Host names only: one label, dots between, 253 bytes total, at least one dot.
DOMAIN_RE = re.compile(r"^(?!-)[a-z0-9-]{1,63}(?<!-)(\.(?!-)[a-z0-9-]{1,63}(?<!-))+$")
AUDIT_USER_AGENT = {"User-Agent": "Aether-house-services/1.0 (agent marketplace; domain-trust-audit)", "Accept": "*/*"}
SEVERITY_COST = {"high": 25, "medium": 10, "low": 3}

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


class DomainIn(BaseModel):
    model_config = ConfigDict(extra="ignore")
    domain: str = Field(min_length=1, max_length=2048)  # a whole URL is accepted; its host is taken


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
    # Second gate, independent of the one in api.py: only the house agent may be paid for
    # work this server performs. Listing is already restricted, but a listing that predates
    # the setting — or any future path that skips that check — must still not earn here.
    if not settings.house_agent_id or service.seller_id != settings.house_agent_id:
        raise _unauthorized(f"service {service_id} is not owned by the house agent")
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


# ------------------------------------------------------- domain-trust-audit: DNS


class DnsAnswer(NamedTuple):
    """One DNS question's outcome, flattened so no check has to know dnspython.

    `status` is "ok" (records returned), "empty" (the name exists but has no
    record of this type), "nxdomain", or "error" (timeout, SERVFAIL, budget)."""

    status: str
    values: tuple[str, ...] = ()
    detail: str = ""


def _rdata_text(rdata: Any) -> str:
    """TXT records arrive as a list of <=255-byte chunks that the publisher
    split for the wire; a long DKIM key is only meaningful rejoined."""
    strings = getattr(rdata, "strings", None)
    if strings is not None:
        return "".join(chunk.decode("utf-8", "replace") for chunk in strings)
    return rdata.to_text()


async def _dns_lookup(qname: str, rdtype: str) -> DnsAnswer:
    """Ask one DNS question. The single seam the tests replace: CI resolves
    differently from this host, so nothing here may depend on live DNS."""
    resolver = dns.asyncresolver.Resolver()
    resolver.timeout = DNS_TIMEOUT_S
    resolver.lifetime = DNS_TIMEOUT_S
    try:
        answer = await resolver.resolve(qname, rdtype, raise_on_no_answer=False)
    except dns.resolver.NXDOMAIN:
        return DnsAnswer("nxdomain")
    except dns.exception.Timeout:
        return DnsAnswer("error", detail="the nameserver did not answer in time")
    except dns.exception.DNSException as exc:
        return DnsAnswer("error", detail=type(exc).__name__)
    if answer.rrset is None:
        return DnsAnswer("empty")
    return DnsAnswer("ok", tuple(_rdata_text(rdata) for rdata in answer.rrset))


class DnsBudget:
    """A hard cap on how many questions one paid call may ask. A domain can
    point its MX at hosts that point at more hosts; the buyer pays $0.10 either
    way, so the work has to stop somewhere."""

    def __init__(self, limit: int) -> None:
        self.left = limit

    async def ask(self, qname: str, rdtype: str) -> DnsAnswer:
        if self.left <= 0:
            return DnsAnswer("error", detail="DNS query budget exhausted")
        self.left -= 1  # spent before awaiting, so concurrent questions cannot overdraw
        return await _dns_lookup(qname, rdtype)


def _mx_hosts(answer: DnsAnswer) -> list[tuple[int, str]]:
    """(preference, host) pairs, lowest preference first. Anything unparseable
    is dropped rather than raised: this is a report, not a parser."""
    hosts = []
    for value in answer.values:
        preference, _, exchange = value.partition(" ")
        exchange = exchange.strip().rstrip(".").lower()
        if exchange and preference.strip().isdigit():
            hosts.append((int(preference), exchange))
    return sorted(hosts)


async def _check_mx(domain: str, mx: DnsAnswer, budget: DnsBudget) -> dict:
    if mx.status == "nxdomain":
        return {"status": "nxdomain", "present": False, "records": [], "note": "the domain itself does not exist"}
    if mx.status == "error":
        return {"status": "error", "present": None, "records": [], "note": mx.detail}
    # RFC 7505: a lone MX whose exchange is the root, "0 .", is a positive
    # statement that the domain receives no mail. `_mx_hosts` drops it, so it is
    # read off the raw answer before that.
    if [value.split()[-1] for value in mx.values if value.split()] == ["."]:
        return {"status": "null_mx", "present": False, "records": ["."], "note": "RFC 7505 null MX: this domain accepts no mail"}
    hosts = _mx_hosts(mx)
    if not hosts:
        return {
            "status": "absent",
            "present": False,
            "records": [],
            "note": "no MX record, so mail would fall back to the A record",
        }
    resolved: dict[str, bool] = {}
    probed = [host for _, host in hosts[:MAX_MX_RESOLVED]]
    for rdtype in ("A", "AAAA"):
        pending = [host for host in probed if not resolved.get(host)]
        if not pending:
            break
        answers = await asyncio.gather(*(budget.ask(host, rdtype) for host in pending))
        for host, answer in zip(pending, answers, strict=True):
            resolved[host] = answer.status == "ok"
    return {
        "status": "ok",
        "present": True,
        "records": [{"preference": preference, "host": host} for preference, host in hosts[:MAX_REPORTED]],
        "probed": probed,
        "resolvable": sorted(host for host in probed if resolved.get(host)),
        "unresolvable": sorted(host for host in probed if not resolved.get(host)),
    }


def _check_spf(txt: DnsAnswer) -> dict:
    if txt.status == "error":
        return {"status": "error", "present": None, "note": txt.detail}
    records = [value for value in txt.values if value.lower().startswith("v=spf1")]
    if not records:
        return {"status": "absent", "present": False, "note": "no v=spf1 TXT record: any host may claim to send as this domain"}
    if len(records) > 1:
        return {
            "status": "multiple",
            "present": True,
            "records": records[:MAX_REPORTED],
            "note": f"{len(records)} SPF records; RFC 7208 makes that a permanent error and receivers ignore all of them",
        }
    record = records[0]
    qualifier, mechanism = None, None
    for token in record.split():
        bare = token[1:] if token[:1] in "+-~?" else token
        if bare.lower() == "all":
            qualifier, mechanism = (token[:1] if token[:1] in "+-~?" else "+"), token
    meaning = {"+": "pass", "-": "fail", "~": "softfail", "?": "neutral"}.get(qualifier or "", "none")
    return {
        "status": "ok",
        "present": True,
        "record": record,
        "all_mechanism": mechanism,
        "all_qualifier": qualifier,
        "policy": meaning,
        "permissive": qualifier in ("+", "?"),
    }


def _dmarc_tags(record: str) -> dict[str, str]:
    tags = {}
    for part in record.split(";"):
        key, _, value = part.partition("=")
        if key.strip():
            tags[key.strip().lower()] = value.strip()
    return tags


def _check_dmarc(txt: DnsAnswer) -> dict:
    if txt.status == "error":
        return {"status": "error", "present": None, "note": txt.detail}
    records = [value for value in txt.values if value.lower().startswith("v=dmarc1")]
    if not records:
        return {"status": "absent", "present": False, "note": "no _dmarc TXT record: receivers are given no policy to apply"}
    if len(records) > 1:
        return {
            "status": "multiple",
            "present": True,
            "records": records[:MAX_REPORTED],
            "note": "more than one DMARC record; receivers ignore all",
        }
    tags = _dmarc_tags(records[0])
    policy = tags.get("p", "").lower()
    return {
        "status": "ok" if policy in ("none", "quarantine", "reject") else "malformed",
        "present": True,
        "record": records[0],
        "policy": policy or None,
        "subdomain_policy": tags.get("sp", "").lower() or None,
        "percent": tags.get("pct") or "100",
        "reporting": bool(tags.get("rua") or tags.get("ruf")),
    }


def _check_dkim(answers: dict[str, DnsAnswer]) -> dict:
    answered = sorted(
        selector
        for selector, answer in answers.items()
        if answer.status == "ok" and any("v=dkim1" in value.lower() or "p=" in value for value in answer.values)
    )
    return {
        "status": "found" if answered else "not_found",
        "selectors_probed": list(DKIM_SELECTORS),
        "selectors_answered": answered,
        "note": (
            "DKIM keys are published under a selector the sender chooses, and selectors cannot be enumerated from DNS. "
            "These are only the common ones: finding none is not proof that the domain does not sign its mail."
        ),
    }


def _check_dnssec(dnskey: DnsAnswer) -> dict:
    if dnskey.status == "error":
        return {"status": "error", "present": None, "note": dnskey.detail}
    present = dnskey.status == "ok" and bool(dnskey.values)
    return {
        "status": "ok",
        "present": present,
        "dnskey_count": len(dnskey.values),
        "note": (
            "presence of a DNSKEY at the zone apex. A full chain of trust also needs a DS record in the parent zone, "
            "which this check does not validate."
        ),
    }


# ------------------------------------------------------- domain-trust-audit: TLS


async def _tls_handshake(address: str, server_hostname: str, *, verify: bool) -> tuple[bytes, str]:
    """Open TLS to one already-approved *address* and hand back (DER certificate,
    negotiated version).

    `address` is a literal IP that `netguard.resolve_public` has passed, so
    `getaddrinfo` here is a no-op format conversion, not a second lookup that
    could answer 127.0.0.1 this time. The hostname is used for SNI and for
    certificate verification only — it is never resolved."""
    context = ssl.create_default_context()
    if not verify:
        # Deliberately unverified, and only ever used as a *second* attempt, to
        # read the subject and dates off a certificate that already failed to
        # verify. Nothing from this handshake is trusted; it is reported as
        # "the chain did not verify" plus the details.
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
    reader, writer = await asyncio.wait_for(
        asyncio.open_connection(host=address, port=TLS_PORT, ssl=context, server_hostname=server_hostname),
        timeout=TLS_TIMEOUT_S,
    )
    try:
        ssl_object = writer.get_extra_info("ssl_object")
        return ssl_object.getpeercert(binary_form=True), ssl_object.version() or "unknown"
    finally:
        writer.close()
        with contextlib.suppress(Exception):  # a peer that vanishes mid-close is not a finding
            await writer.wait_closed()


def _hostname_matches(domain: str, names: list[str]) -> bool:
    """RFC 6125 in the only shape that matters here: exact, or a wildcard that
    covers exactly one leading label. `ssl.match_hostname` went away in 3.12."""
    for name in names:
        candidate = name.lower().rstrip(".")
        if candidate == domain:
            return True
        if candidate.startswith("*.") and "." in domain and domain.split(".", 1)[1] == candidate[2:]:
            return True
    return False


def _cert_names(certificate: x509.Certificate) -> list[str]:
    try:
        san = certificate.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
        names = list(san.get_values_for_type(x509.DNSName))
    except x509.ExtensionNotFound:
        names = []
    if not names:  # pre-2017 style: fall back to the common name
        names = [attribute.value for attribute in certificate.subject.get_attributes_for_oid(x509.oid.NameOID.COMMON_NAME)]
    return [name for name in names if isinstance(name, str)]


def _describe_certificate(der: bytes, domain: str, tls_version: str, verify_error: str | None) -> dict:
    certificate = x509.load_der_x509_certificate(der)
    not_before, not_after = certificate.not_valid_before_utc, certificate.not_valid_after_utc
    now = datetime.now(UTC)
    names = _cert_names(certificate)
    return {
        "status": "ok",
        "reachable": True,
        "tls_version": tls_version,
        "chain_trusted": verify_error is None,
        "chain_error": verify_error,
        "subject": certificate.subject.rfc4514_string(),
        "issuer": certificate.issuer.rfc4514_string(),
        "serial": format(certificate.serial_number, "x"),
        "not_before": not_before.isoformat().replace("+00:00", "Z"),
        "not_after": not_after.isoformat().replace("+00:00", "Z"),
        "days_until_expiry": (not_after - now).days,
        "expired": not_after <= now,
        "not_yet_valid": not_before > now,
        "hostname_match": _hostname_matches(domain, names),
        "names": names[:MAX_REPORTED],
    }


async def _check_tls(domain: str, addresses: list[str]) -> dict:
    """Certificate and protocol facts for https://<domain>.

    `addresses` are the ones `netguard.resolve_public` already approved, and
    they are the only ones connected to. A dead first address is retried
    against the next, because that is a round-robin domain, not a finding."""
    failure = {"status": "unreachable", "reachable": False, "error": "no address to connect to"}
    for address in addresses[:MAX_TLS_ADDRESSES]:
        verify_error: str | None = None
        try:
            der, version = await _tls_handshake(address, domain, verify=True)
        except ssl.SSLCertVerificationError as exc:
            verify_error = exc.verify_message or str(exc)
            try:  # the certificate is still worth reporting — that is the whole finding
                der, version = await _tls_handshake(address, domain, verify=False)
            except (OSError, ssl.SSLError) as retry:
                return {"status": "handshake_failed", "reachable": True, "error": _tls_error(retry), "chain_error": verify_error}
        except ssl.SSLError as exc:  # spoke TCP, would not speak TLS: a finding, not another address
            return {"status": "handshake_failed", "reachable": True, "error": _tls_error(exc)}
        except OSError as exc:  # includes TimeoutError from wait_for
            failure = {"status": "unreachable", "reachable": False, "error": _tls_error(exc), "address": address}
            continue
        try:
            return _describe_certificate(der, domain, version, verify_error) | {"address": address}
        except ValueError as exc:  # spoke TLS, then sent something that is not a certificate
            return {"status": "unreadable_certificate", "reachable": True, "error": str(exc), "chain_error": verify_error}
    return failure


def _tls_error(exc: BaseException) -> str:
    if isinstance(exc, asyncio.TimeoutError):
        return f"no TLS handshake within {TLS_TIMEOUT_S:.0f}s"
    return f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__


# --------------------------------------------------- domain-trust-audit: headers


async def _check_headers(http: httpx.AsyncClient, domain: str) -> dict:
    """The response headers of https://<domain>/.

    A bare domain very often 301s to www, and the security headers that matter
    are on the page the visitor lands on, so a couple of hops are followed —
    each one re-checked against netguard first, exactly like `_fetch_page`."""
    url = f"https://{domain}/"
    for _ in range(MAX_HEADER_REDIRECTS + 1):
        try:
            # https only, on purpose: a hop that drops to http is not the page a
            # visitor gets over TLS, so it is refused rather than followed.
            await netguard.check_public_url(url)
        except netguard.UnsafeURL as exc:
            return {"status": "refused", "url": url, "error": f"url {exc}"}
        except ValueError:  # urlparse rejects a malformed Location outright
            return {"status": "refused", "url": url, "error": "url is malformed"}
        try:
            async with http.stream("GET", url, timeout=HEADER_TIMEOUT, follow_redirects=False, headers=AUDIT_USER_AGENT) as resp:
                if resp.status_code in (301, 302, 303, 307, 308) and resp.headers.get("location"):
                    url = urljoin(str(resp.url), resp.headers["location"])
                    continue
                read = 0
                async for chunk in resp.aiter_bytes():  # drained so the connection can be reused, never kept
                    read += len(chunk)
                    if read > MAX_HEADER_BYTES:
                        break
                return {
                    "status": "ok",
                    "url": str(resp.url),
                    "http_status": resp.status_code,
                    "headers": {
                        name: {"present": name in resp.headers, "value": resp.headers.get(name)} for name in SECURITY_HEADERS
                    },
                    "missing": [name for name in SECURITY_HEADERS if name not in resp.headers],
                }
        except httpx.TimeoutException:
            return {"status": "timeout", "url": url, "error": "the site did not answer in time"}
        except (httpx.HTTPError, httpx.InvalidURL) as exc:  # InvalidURL is not an HTTPError
            return {"status": "unreachable", "url": url, "error": type(exc).__name__}
    return {"status": "too_many_redirects", "url": url, "error": f"more than {MAX_HEADER_REDIRECTS} redirects"}


# -------------------------------------------------- domain-trust-audit: scoring


def _findings(checks: dict) -> list[dict]:
    """Turn the raw checks into the sentences a buyer actually wants. Severity
    is about mail and identity: anything that lets a stranger send as this
    domain, or breaks TLS identity, is high."""
    out: list[dict] = []

    def add(severity: str, check: str, message: str) -> None:
        out.append({"severity": severity, "check": check, "message": message})

    mx = checks["mx"]
    if mx["status"] == "nxdomain":
        add("high", "mx", "the domain does not exist in DNS")
    elif mx["status"] == "absent":
        add("high", "mx", "no MX record: this domain cannot reliably receive mail")
    elif mx["status"] == "null_mx":
        add("low", "mx", "a null MX declares that this domain accepts no mail; that is deliberate, not a fault")
    elif mx["status"] == "ok" and not mx["resolvable"]:
        add("high", "mx", f"none of the mail hosts resolve to an address: {', '.join(mx['unresolvable'])}")
    elif mx["status"] == "ok" and mx["unresolvable"]:
        add("medium", "mx", f"some mail hosts do not resolve: {', '.join(mx['unresolvable'])}")

    spf = checks["spf"]
    if spf["status"] == "absent":
        add("high", "spf", "no SPF record: nothing states which hosts may send as this domain")
    elif spf["status"] == "multiple":
        add("high", "spf", "more than one SPF record, which receivers treat as a permanent error and ignore")
    elif spf["status"] == "ok":
        if spf["all_qualifier"] == "+":
            add("high", "spf", "SPF ends in +all, which authorises every host on the internet to send as this domain")
        elif spf["all_qualifier"] == "?":
            add("medium", "spf", "SPF ends in ?all (neutral), which asserts nothing and gives receivers no reason to reject")
        elif spf["all_qualifier"] is None:
            add("medium", "spf", "SPF has no 'all' mechanism, so unlisted senders fall through with no result")

    dmarc = checks["dmarc"]
    if dmarc["status"] == "absent":
        add("high", "dmarc", "no DMARC record: receivers are told nothing about what to do with forged mail")
    elif dmarc["status"] == "multiple":
        add("high", "dmarc", "more than one DMARC record, which receivers ignore entirely")
    elif dmarc["status"] == "malformed":
        add("high", "dmarc", "the DMARC record has no usable p= policy")
    elif dmarc["status"] == "ok":
        if dmarc["policy"] == "none":
            add("medium", "dmarc", "DMARC is p=none: it monitors only and does not ask receivers to act on forgeries")
        elif dmarc["policy"] == "quarantine":
            add("low", "dmarc", "DMARC is p=quarantine; p=reject is the end state")
        if not dmarc["reporting"]:
            add("low", "dmarc", "DMARC has no rua= address, so no aggregate reports come back")

    if checks["dkim"]["status"] == "not_found":
        add("low", "dkim", "none of the common DKIM selectors answered; this does not prove the domain is unsigned")
    if checks["dnssec"].get("present") is False:
        add("low", "dnssec", "no DNSKEY at the apex: this zone's answers are not signed")

    tls = checks["tls"]
    if tls["status"] == "unreachable":
        add("medium", "tls", f"nothing answered TLS on port {TLS_PORT}: {tls.get('error')}")
    elif tls["status"] in ("handshake_failed", "unreadable_certificate"):
        add("high", "tls", f"the TLS handshake did not complete: {tls.get('error')}")
    elif tls["status"] == "ok":
        if tls["expired"]:
            add("high", "tls", f"the certificate expired on {tls['not_after']}")
        elif tls["days_until_expiry"] < CERT_EXPIRY_WARN_DAYS:
            add("medium", "tls", f"the certificate expires in {tls['days_until_expiry']} days, on {tls['not_after']}")
        if tls["not_yet_valid"]:
            add("high", "tls", f"the certificate is not valid until {tls['not_before']}")
        if not tls["hostname_match"]:
            add("high", "tls", "the certificate does not name this domain")
        if not tls["chain_trusted"]:
            add("high", "tls", f"the certificate chain does not verify: {tls['chain_error']}")
        if tls["tls_version"] in ("SSLv3", "TLSv1", "TLSv1.1"):
            add("medium", "tls", f"the server negotiated {tls['tls_version']}, which is deprecated")

    headers = checks["headers"]
    if headers["status"] == "ok":
        for name in headers["missing"]:
            severity = "medium" if name == "strict-transport-security" else "low"
            add(severity, "headers", f"{name} is not set")
    elif headers["status"] == "refused":
        add("high", "headers", f"the site could not be fetched: {headers['error']}")
    elif headers["status"] != "skipped":
        add("low", "headers", f"the site's headers could not be read: {headers.get('error', headers['status'])}")

    return out


def _verdict(findings: list[dict]) -> tuple[str, int]:
    score = max(0, 100 - sum(SEVERITY_COST.get(finding["severity"], 0) for finding in findings))
    severities = {finding["severity"] for finding in findings}
    if "high" in severities:
        return "fail", score
    if "medium" in severities:
        return "warn", score
    return "pass", score


# ---------------------------------------------- domain-trust-audit: the endpoint


def _normalise_domain(value: str) -> str:
    """A bare domain, or the host of a URL. Returns a lowercase ASCII host.

    An IP literal is handed back unchanged rather than rejected here: whether
    an address may be connected to is netguard's decision, not this parser's,
    and routing it there keeps one rule in one place."""
    raw = value.strip()
    try:
        parsed = urlparse(raw if "://" in raw else f"//{raw}", scheme="https")
        host = parsed.hostname or ""
    except ValueError:
        raise _error(422, "invalid_domain", "domain is not a hostname or URL") from None
    host = host.strip().rstrip(".").lower()
    if not host:
        raise _error(422, "invalid_domain", "domain is empty")
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        return host
    try:
        host = host.encode("idna").decode("ascii")
    except (UnicodeError, UnicodeDecodeError):
        raise _error(422, "invalid_domain", f"{value[:80]!r} is not a valid international domain name") from None
    if len(host) > 253 or not DOMAIN_RE.match(host):
        raise _error(422, "invalid_domain", "domain must be a hostname such as example.com")
    return host


def _skipped(reason: str) -> dict:
    return {"status": "skipped", "reason": reason}


async def _bounded(coro, seconds: float, on_timeout: dict) -> dict:
    """Run one check under a hard ceiling, reporting rather than raising.

    The checks own their own errors; this owns only the clock. A peer that
    trickles bytes gets `seconds` and no more, and the check it starved is
    reported as a result so the buyer still receives an answer for every row.
    A budget already spent is the same outcome, without opening the socket."""
    if seconds <= 0:
        coro.close()
        return on_timeout
    try:
        return await asyncio.wait_for(coro, seconds)
    except (TimeoutError, asyncio.TimeoutError):
        return on_timeout


@router.post("/domain-trust-audit")
async def domain_trust_audit(request: Request):
    """Mail authentication, TLS and HTTP security-header posture for one domain.

    Everything here is observable from the public internet with free queries.
    No check can fail the call: an absent record, a dead mail host and a domain
    that does not exist at all are all reported as results."""
    body = _parse(DomainIn, await authorized_input(request, "domain-trust-audit"))
    domain = _normalise_domain(body.domain)
    deadline = time.monotonic() + AUDIT_BUDGET_S

    # One resolution, before any connection, and the addresses it approves are the
    # only ones the TLS socket may use. A domain that points anywhere inside our
    # own networks is refused outright; one that resolves nowhere is a finding.
    unreachable: str | None = None
    addresses: list[str] = []
    try:
        addresses = await netguard.resolve_public(domain, TLS_PORT)
    except netguard.UnresolvableHost:
        unreachable = f"{domain} has no address record"
    except netguard.UnsafeURL as exc:
        raise _error(422, "refused_domain", f"domain {exc}") from None

    budget = DnsBudget(MAX_DNS_QUERIES)
    questions = [(domain, "MX"), (domain, "TXT"), (domain, "DNSKEY"), (f"_dmarc.{domain}", "TXT")]
    questions += [(f"{selector}._domainkey.{domain}", "TXT") for selector in DKIM_SELECTORS]
    mx_answer, txt, dnskey, dmarc_txt, *dkim_answers = await asyncio.gather(
        *(budget.ask(qname, rdtype) for qname, rdtype in questions)
    )

    checks = {
        "mx": await _check_mx(domain, mx_answer, budget),
        "spf": _check_spf(txt),
        "dmarc": _check_dmarc(dmarc_txt),
        "dkim": _check_dkim(dict(zip(DKIM_SELECTORS, dkim_answers, strict=True))),
        "dnssec": _check_dnssec(dnskey),
    }

    if unreachable is not None:
        checks["tls"] = {"status": "unreachable", "reachable": False, "error": unreachable}
        checks["headers"] = {"status": "unreachable", "url": f"https://{domain}/", "error": unreachable}
    else:
        # Whatever the individual timeouts do, the two connections together get
        # what is left of the call's budget and not a second more: a server that
        # trickles bytes must not be able to hold a worker open.
        left = min(CONNECT_PHASE_S, deadline - time.monotonic())
        spent = _skipped("the call's time budget was spent on DNS")
        checks["tls"], checks["headers"] = await asyncio.gather(
            _bounded(_check_tls(domain, addresses), left, spent),
            _bounded(_check_headers(request.app.state.http, domain), left, spent),
        )

    findings = _findings(checks)
    verdict, score = _verdict(findings)
    return {
        "output": {
            "domain": domain,
            "verdict": verdict,
            "score": score,
            "findings": findings,
            "checks": checks,
            "resolved_addresses": addresses,
            "checked_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "dns_queries_used": MAX_DNS_QUERIES - budget.left,
        }
    }
