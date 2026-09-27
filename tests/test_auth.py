import jwt

from .helpers import new_agent


async def test_client_credentials_grant_and_claims(client):
    reg = (await client.post("/v1/agents", json={"name": "a", "scopes": ["buy_inference"]})).json()
    r = await client.post(
        "/oauth/token",
        data={"grant_type": "client_credentials", "client_id": reg["client_id"], "client_secret": reg["client_secret"]},
    )
    assert r.status_code == 200 and r.headers["cache-control"] == "no-store"
    body = r.json()
    assert body["token_type"] == "Bearer" and body["scope"] == "buy_inference"

    jwks = (await client.get("/.well-known/jwks.json")).json()["keys"]
    header = jwt.get_unverified_header(body["access_token"])
    key = next(k for k in jwks if k["kid"] == header["kid"])
    claims = jwt.decode(body["access_token"], jwt.PyJWK(key).key, algorithms=["RS256"], audience="aether-clearinghouse")
    assert claims["sub"] == reg["agent_id"] and claims["typ"] == "access"
    assert claims["exp"] - claims["iat"] == body["expires_in"]


async def test_bad_credentials_and_scopes(client):
    reg = (await client.post("/v1/agents", json={"name": "a", "scopes": ["buy_inference"]})).json()
    r = await client.post("/oauth/token", data={"grant_type": "client_credentials"}, auth=(reg["client_id"], "wrong"))
    assert r.status_code == 401 and r.json()["error"] == "invalid_client"
    r = await client.post(
        "/oauth/token",
        data={"grant_type": "client_credentials", "scope": "sell_compute"},
        auth=(reg["client_id"], reg["client_secret"]),
    )
    assert r.status_code == 400 and r.json()["error"] == "invalid_scope"
    r = await client.post("/oauth/token", data={"grant_type": "password"}, auth=(reg["client_id"], reg["client_secret"]))
    assert r.json()["error"] == "unsupported_grant_type"


async def test_revocation(client):
    agent = await new_agent(client, "a", ["buy_inference"])
    assert (await client.get("/v1/agents/me", headers=agent["headers"])).status_code == 200
    assert (await client.post("/oauth/revoke", headers=agent["headers"])).status_code == 200
    r = await client.get("/v1/agents/me", headers=agent["headers"])
    assert r.status_code == 401 and r.json()["detail"]["error_description"] == "token revoked"


async def test_missing_and_forged_tokens(client):
    assert (await client.get("/v1/agents/me")).status_code == 401
    forged = jwt.encode(
        {"sub": "agt_x", "aud": "aether-clearinghouse"}, "attacker-chosen-hmac-secret-0123456789", algorithm="HS256"
    )
    assert (await client.get("/v1/agents/me", headers={"Authorization": f"Bearer {forged}"})).status_code == 401
