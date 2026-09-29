"""Address rules for every outbound connection the server makes on behalf of
an agent (seller endpoints, buyer-supplied page URLs).

One module, because SSRF is the whole risk: an agent-supplied URL must never
reach the host's own networks. Note that the five "not public" flags are not
sufficient on current CPython — 100.64.0.0/10, the carrier-grade NAT range
Tailscale hands out, is neither `is_private` nor `is_global` on 3.13 — so an
address has to be positively `is_global` to be accepted.
"""

import asyncio
import ipaddress
import socket
from urllib.parse import urlparse

IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address


class UnsafeURL(ValueError):
    """A URL the server refuses to connect to."""


def ip_is_public(ip: IPAddress) -> bool:
    if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast or ip.is_unspecified:
        return False
    return ip.is_global


async def resolve_public(host: str, port: int) -> list[str]:
    """Resolve `host` and refuse it unless *every* address it answers with is
    public: one private record is all an attacker needs."""
    try:
        infos = await asyncio.get_running_loop().getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror:
        raise UnsafeURL("host does not resolve") from None
    addresses = []
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if not ip_is_public(ip):
            raise UnsafeURL(f"resolves to the non-public address {ip}")
        addresses.append(str(ip))
    if not addresses:
        raise UnsafeURL("host does not resolve")
    return addresses


async def check_public_url(url: str, *, require_https: bool = True) -> str:
    """Resolve-then-check a URL. Returns its hostname; raises `UnsafeURL`.

    Call this again for every redirect hop: the first URL being public says
    nothing about where it points."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise UnsafeURL("must be an absolute http(s) URL")
    if require_https and parsed.scheme != "https":
        raise UnsafeURL("must use https")
    if parsed.username or parsed.password:
        raise UnsafeURL("must not carry credentials")
    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError:
        raise UnsafeURL("has an invalid port") from None
    await resolve_public(parsed.hostname, port)
    return parsed.hostname
