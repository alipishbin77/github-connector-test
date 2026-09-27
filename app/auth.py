"""M2M identity: OAuth2 Client Credentials grant (RFC 6749 §4.4) issuing
short-lived RS256 JWT access tokens (RFC 9068 style), plus the per-request
"delivery tokens" the proxy router presents to seller endpoints.

Agents never see each other's credentials. Sellers verify delivery tokens
offline against the public JWKS at /.well-known/jwks.json.
"""

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import time
import uuid
from dataclasses import dataclass

import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import APIRouter, Depends, Form, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.security import (
    HTTPAuthorizationCredentials,
    HTTPBasic,
    HTTPBasicCredentials,
    HTTPBearer,
)
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .config import settings
from .db import SessionLocal, get_session
from .models import Agent

log = logging.getLogger("aether.auth")

SCOPE_BUY = "buy_inference"
SCOPE_SELL = "sell_compute"
ALL_SCOPES = frozenset({SCOPE_BUY, SCOPE_SELL})
ACCESS_AUDIENCE = "aether-clearinghouse"
ALGORITHM = "RS256"


def delivery_audience(seller_id: str) -> str:
    return f"aether-seller:{seller_id}"


def _b64u(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _b64u_uint(value: int) -> str:
    return _b64u(value.to_bytes((value.bit_length() + 7) // 8, "big"))


# --------------------------------------------------------------------------- keys


class SigningKey:
    def __init__(self, private_key: rsa.RSAPrivateKey):
        self.private_key = private_key
        self.public_key = private_key.public_key()
        numbers = self.public_key.public_numbers()
        core = {"e": _b64u_uint(numbers.e), "kty": "RSA", "n": _b64u_uint(numbers.n)}
        # RFC 7638 thumbprint as key id
        self.kid = _b64u(hashlib.sha256(json.dumps(core, separators=(",", ":"), sort_keys=True).encode()).digest())
        self.jwk = {**core, "use": "sig", "alg": ALGORITHM, "kid": self.kid}

    @classmethod
    def load_or_create(cls, pem: str | None = None, path: str | None = None) -> "SigningKey":
        if pem:
            return cls(serialization.load_pem_private_key(pem.encode(), password=None))
        if path:
            try:
                with open(path, "rb") as fh:
                    return cls(serialization.load_pem_private_key(fh.read(), password=None))
            except FileNotFoundError:
                pass
            key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
            data = key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
            try:
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError:  # another worker won the race
                return cls.load_or_create(path=path)
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
            log.info("generated new RS256 signing key at %s", path)
            return cls(key)
        log.warning("no JWT key configured; using an ephemeral key (single-process only)")
        return cls(rsa.generate_private_key(public_exponent=65537, key_size=2048))


_signing_key: SigningKey | None = None


def init_signing_key() -> SigningKey:
    global _signing_key
    if _signing_key is None:
        _signing_key = SigningKey.load_or_create(settings.jwt_private_key_pem, settings.jwt_key_file)
    return _signing_key


def signing_key() -> SigningKey:
    return init_signing_key()


# ------------------------------------------------------------------ client secrets

_SCRYPT_N, _SCRYPT_R, _SCRYPT_P = 2**14, 8, 1


def generate_client_credentials() -> tuple[str, str]:
    return f"agt_cli_{secrets.token_hex(12)}", f"aes_{secrets.token_urlsafe(32)}"


def hash_secret(secret: str) -> str:
    salt = os.urandom(16)
    digest = hashlib.scrypt(secret.encode(), salt=salt, n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P, dklen=32)
    return f"scrypt${_SCRYPT_N}${_SCRYPT_R}${_SCRYPT_P}${_b64u(salt)}${_b64u(digest)}"


_DUMMY_HASH: str | None = None


def _dummy_hash() -> str:
    global _DUMMY_HASH
    if _DUMMY_HASH is None:
        _DUMMY_HASH = hash_secret(secrets.token_urlsafe(16))
    return _DUMMY_HASH


def verify_secret(secret: str, encoded: str) -> bool:
    try:
        algo, n, r, p, salt, digest = encoded.split("$")
    except ValueError:
        return False
    if algo != "scrypt":
        return False
    pad = lambda s: s + "=" * (-len(s) % 4)  # noqa: E731
    expected = base64.urlsafe_b64decode(pad(digest))
    actual = hashlib.scrypt(
        secret.encode(),
        salt=base64.urlsafe_b64decode(pad(salt)),
        n=int(n),
        r=int(r),
        p=int(p),
        dklen=len(expected),
    )
    return hmac.compare_digest(actual, expected)


# ------------------------------------------------------------------------ tokens


@dataclass(frozen=True)
class AuthContext:
    agent_id: str
    client_id: str
    scopes: frozenset[str]
    jti: str
    exp: int


def issue_access_token(agent: Agent, scopes: set[str]) -> tuple[str, int, dict]:
    now = int(time.time())
    ttl = settings.access_token_ttl_s
    claims = {
        "iss": settings.jwt_issuer,
        "sub": agent.id,
        "aud": ACCESS_AUDIENCE,
        "iat": now,
        "nbf": now,
        "exp": now + ttl,
        "jti": uuid.uuid4().hex,
        "client_id": agent.client_id,
        "scope": " ".join(sorted(scopes)),
        "tv": agent.token_version,
        "typ": "access",
    }
    key = signing_key()
    token = jwt.encode(claims, key.private_key, algorithm=ALGORITHM, headers={"kid": key.kid, "typ": "at+jwt"})
    return token, ttl, claims


def issue_delivery_token(
    *,
    seller_id: str,
    job_id: str,
    trade_id: str,
    instrument: str,
    max_tokens: int,
    resume_from: int,
    body_sha256: str,
) -> str:
    """Single-use, seconds-lived capability for exactly one seller request.
    Bound to the request body by hash so it cannot be replayed with a
    different prompt or a larger max_tokens."""
    now = int(time.time())
    claims = {
        "iss": settings.jwt_issuer,
        "sub": job_id,
        "aud": delivery_audience(seller_id),
        "iat": now,
        "nbf": now,
        "exp": now + settings.delivery_token_ttl_s,
        "jti": uuid.uuid4().hex,
        "typ": "delivery",
        "trade_id": trade_id,
        "instrument": instrument,
        "max_tokens": max_tokens,
        "resume_from": resume_from,
        "body_sha256": body_sha256,
    }
    key = signing_key()
    return jwt.encode(claims, key.private_key, algorithm=ALGORITHM, headers={"kid": key.kid})


def _unauthorized(error: str, description: str) -> HTTPException:
    return HTTPException(
        status.HTTP_401_UNAUTHORIZED,
        detail={"error": error, "error_description": description},
        headers={"WWW-Authenticate": f'Bearer realm="aether", error="{error}"'},
    )


def decode_access_token(token: str) -> dict:
    try:
        claims = jwt.decode(
            token,
            signing_key().public_key,
            algorithms=[ALGORITHM],
            audience=ACCESS_AUDIENCE,
            issuer=settings.jwt_issuer,
            leeway=5,
            options={"require": ["exp", "iat", "nbf", "iss", "aud", "sub", "jti"]},
        )
    except jwt.ExpiredSignatureError:
        raise _unauthorized("invalid_token", "token expired") from None
    except jwt.InvalidTokenError as exc:
        raise _unauthorized("invalid_token", f"token rejected: {exc}") from None
    if claims.get("typ") != "access":
        raise _unauthorized("invalid_token", "not an access token")
    return claims


def revoked_key(jti: str) -> str:
    return f"{settings.redis_prefix}:revoked:{jti}"


_bearer = HTTPBearer(auto_error=False)


async def authenticate(
    request: Request,
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> AuthContext:
    if creds is None:
        raise _unauthorized("invalid_request", "missing bearer token")
    claims = decode_access_token(creds.credentials)
    if await request.app.state.redis.exists(revoked_key(claims["jti"])):
        raise _unauthorized("invalid_token", "token revoked")
    # Own short-lived session: a request-scoped one would stay checked out for
    # the whole duration of a streaming inference response.
    async with SessionLocal() as session:
        agent = await session.get(Agent, claims["sub"])
    if agent is None or not agent.is_active or agent.token_version != claims.get("tv"):
        raise _unauthorized("invalid_token", "agent disabled or token generation revoked")
    return AuthContext(
        agent_id=agent.id,
        client_id=claims["client_id"],
        scopes=frozenset(claims.get("scope", "").split()),
        jti=claims["jti"],
        exp=claims["exp"],
    )


def check_scopes(ctx: AuthContext, *required: str) -> None:
    missing = set(required) - ctx.scopes
    if missing:
        needed = " ".join(sorted(required))
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            detail={"error": "insufficient_scope", "error_description": f"requires scope: {needed}"},
            headers={"WWW-Authenticate": f'Bearer realm="aether", error="insufficient_scope", scope="{needed}"'},
        )


def require_scopes(*required: str):
    async def dependency(ctx: AuthContext = Depends(authenticate)) -> AuthContext:
        check_scopes(ctx, *required)
        return ctx

    return dependency


# ------------------------------------------------------------------------ routes

router = APIRouter(tags=["oauth2"])
_basic = HTTPBasic(auto_error=False)


def _oauth_error(code: int, error: str, description: str) -> JSONResponse:
    headers = {"Cache-Control": "no-store"}
    if code == 401:
        headers["WWW-Authenticate"] = 'Basic realm="aether"'
    return JSONResponse({"error": error, "error_description": description}, status_code=code, headers=headers)


@router.post("/oauth/token")
async def token_endpoint(
    grant_type: str = Form(...),
    client_id: str | None = Form(None),
    client_secret: str | None = Form(None),
    scope: str | None = Form(None),
    basic: HTTPBasicCredentials | None = Depends(_basic),
    session: AsyncSession = Depends(get_session),
):
    if grant_type != "client_credentials":
        return _oauth_error(400, "unsupported_grant_type", "only client_credentials is supported")
    if basic is not None:  # RFC 6749 §2.3.1: HTTP Basic takes precedence
        client_id, client_secret = basic.username, basic.password
    if not client_id or not client_secret:
        return _oauth_error(401, "invalid_client", "client authentication required")

    agent = (await session.execute(select(Agent).where(Agent.client_id == client_id))).scalar_one_or_none()
    # Always spend the scrypt cost so response time does not reveal valid client_ids.
    encoded = agent.client_secret_hash if agent else _dummy_hash()
    valid = await asyncio.to_thread(verify_secret, client_secret, encoded)
    if agent is None or not valid or not agent.is_active:
        return _oauth_error(401, "invalid_client", "client authentication failed")

    requested = set(scope.split()) if scope else agent.scopes
    if not requested <= agent.scopes:
        return _oauth_error(400, "invalid_scope", f"allowed scopes: {' '.join(sorted(agent.scopes))}")

    token, ttl, _ = issue_access_token(agent, requested)
    log.info("issued access token agent=%s scope=%s ttl=%ss", agent.id, " ".join(sorted(requested)), ttl)
    return JSONResponse(
        {"access_token": token, "token_type": "Bearer", "expires_in": ttl, "scope": " ".join(sorted(requested))},
        headers={"Cache-Control": "no-store", "Pragma": "no-cache"},
    )


@router.post("/oauth/revoke")
async def revoke_endpoint(request: Request, ctx: AuthContext = Depends(authenticate)):
    """Revoke the presented access token (RFC 7009 subset)."""
    ttl = max(1, ctx.exp - int(time.time()))
    await request.app.state.redis.set(revoked_key(ctx.jti), "1", ex=ttl)
    return {"revoked": ctx.jti}


@router.get("/.well-known/jwks.json")
async def jwks():
    return {"keys": [signing_key().jwk]}


@router.get("/.well-known/oauth-authorization-server")
async def authorization_server_metadata():
    return {
        "issuer": settings.jwt_issuer,
        "token_endpoint": "/oauth/token",
        "revocation_endpoint": "/oauth/revoke",
        "jwks_uri": "/.well-known/jwks.json",
        "grant_types_supported": ["client_credentials"],
        "token_endpoint_auth_methods_supported": ["client_secret_basic", "client_secret_post"],
        "scopes_supported": sorted(ALL_SCOPES),
    }
