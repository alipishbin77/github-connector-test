"""Mock Buyer Agent: the end-to-end M2M negotiation.

  1. register + OAuth2 client_credentials -> short-lived JWT (buy_inference)
  2. show that the M2M boundary holds: wrong scope -> 403, no escrow -> 402
  3. fund the wallet (sandbox faucet)
  4. read the order book, send a market bid with a price cap (escrowed first)
  5. stream several concurrent inference jobs through the proxy router,
     surviving simulated spot preemption via checkpoint-resume / failover
  6. release unused allocations and print the micro-transaction ledger + audit

Run:  python -m app.buyer_agent_sim
"""

import asyncio
import base64
import json
import os
import sys
import time

import httpx

CH = os.environ.get("CLEARINGHOUSE_URL", "http://localhost:8000").rstrip("/")
INSTRUMENT = os.environ.get("INSTRUMENT", "llama-3.1-70b-instruct")
BUY_TOKENS = int(os.environ.get("BUY_TOKENS", "1000"))
MAX_PRICE = os.environ.get("MAX_PRICE_USD_PER_MTOK", "0.50")
JOBS = int(os.environ.get("JOBS", "3"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "150"))
FAUCET_USD = os.environ.get("FAUCET_USD", "1.00")
MIN_BOOK_TOKENS = int(os.environ.get("MIN_BOOK_TOKENS", str(BUY_TOKENS)))

COLORS = ["\033[36m", "\033[35m", "\033[33m", "\033[32m", "\033[34m"]
RESET, BOLD, DIM, RED, GREEN = "\033[0m", "\033[1m", "\033[2m", "\033[31m", "\033[32m"
if not sys.stdout.isatty() and not os.environ.get("FORCE_COLOR"):
    COLORS = [""] * 5
    RESET = BOLD = DIM = RED = GREEN = ""


def step(title: str) -> None:
    print(f"\n{BOLD}== {title} {'=' * max(0, 70 - len(title))}{RESET}", flush=True)


def jwt_claims(token: str) -> dict:
    payload = token.split(".")[1]
    return json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))


async def stream_job(client: httpx.AsyncClient, headers: dict, n: int, prompt: str) -> dict:
    color = COLORS[n % len(COLORS)]
    tag = f"{color}[job{n}]{RESET}"
    text: list[str] = []
    result: dict = {}
    event = None
    async with client.stream(
        "POST",
        "/v1/inference",
        json={"instrument": INSTRUMENT, "prompt": prompt, "max_tokens": MAX_TOKENS},
        headers=headers,
        timeout=httpx.Timeout(60, read=60),
    ) as resp:
        if resp.status_code != 200:
            body = await resp.aread()
            print(f"{tag} {RED}HTTP {resp.status_code}: {body.decode()}{RESET}")
            return {"error": resp.status_code}
        async for line in resp.aiter_lines():
            if line.startswith("event:"):
                event = line[6:].strip()
                continue
            if not line.startswith("data:"):
                continue
            data = json.loads(line[5:])
            if event == "meta":
                print(
                    f"{tag} routed job={data['job_id']} -> seller={data['seller_id'][:16]}… "
                    f"trade={data['trade_id'][:24]}… @ ${data['price_usd_per_mtok']}/1M reserved={data['reserved_tokens']}"
                )
            elif event == "resume":
                kind = "FAILOVER to" if data["failover"] else "RESUME on"
                print(f"{tag} {RED}interrupted: {data['reason']}{RESET}")
                print(f"{tag} {BOLD}{kind} seller={data['seller_id'][:16]}… from checkpoint token #{data['resume_from']}{RESET}")
            elif event in ("done", "error"):
                result = data | {"event": event}
            else:
                text.append(data["token"])
                if len(text) % 50 == 0:
                    print(f"{tag} {DIM}…{len(text)} tokens streamed{RESET}")
            event = None
    full = "".join(text)
    status = f"{GREEN}done{RESET}" if result.get("event") == "done" else f"{RED}error{RESET}"
    print(
        f"{tag} {status}: {result.get('tokens')} tokens, cost {result.get('cost_usd')}, attempts={result.get('attempts')}, "
        f"failovers={result.get('failovers')}, finish={result.get('finish_reason') or result.get('error')}"
    )
    for seg in result.get("segments", []):
        print(f"{tag}   segment seller={seg['seller_id'][:16]}… trade={seg['trade_id'][:24]}… tokens={seg['tokens']}")
    print(f"{tag} {DIM}text: {full[:140]}{'…' if len(full) > 140 else ''}{RESET}")
    result["text"] = full
    return result


async def main() -> int:
    async with httpx.AsyncClient(base_url=CH, timeout=30) as c:
        step("1. M2M identity: register + OAuth2 client_credentials grant")
        reg = (await c.post("/v1/agents", json={"name": "buyer-sim", "scopes": ["buy_inference"]})).raise_for_status().json()
        print(f"registered agent_id={reg['agent_id']} client_id={reg['client_id']} (secret shown once, stored as scrypt hash)")
        tok = (
            (
                await c.post(
                    "/oauth/token",
                    data={"grant_type": "client_credentials", "scope": "buy_inference"},
                    auth=(reg["client_id"], reg["client_secret"]),
                )
            )
            .raise_for_status()
            .json()
        )
        claims = jwt_claims(tok["access_token"])
        print(f"access_token: RS256 JWT, expires_in={tok['expires_in']}s")
        print(json.dumps({k: claims[k] for k in ("iss", "sub", "aud", "scope", "exp", "jti", "typ", "tv")}, indent=2))
        H = {"Authorization": f"Bearer {tok['access_token']}"}

        step("2. The M2M boundary holds")
        r = await c.post(
            "/v1/orders",
            headers=H,
            json={"instrument": INSTRUMENT, "side": "ask", "price_usd_per_mtok": "0.10", "quantity_tokens": 10},
        )
        print(
            f"sell attempt with a buy_inference token   -> HTTP {r.status_code} {r.json()['detail']['error']} "
            f"({r.headers.get('www-authenticate')})"
        )
        r = await c.post("/v1/inference", headers=H, json={"instrument": INSTRUMENT, "prompt": "hi", "max_tokens": 10})
        print(f"inference before escrowing any funds      -> HTTP {r.status_code} {r.json()['detail']['error']}")
        r = await c.get("/v1/agents/me", headers={"Authorization": "Bearer " + tok["access_token"][:-4] + "AAAA"})
        print(f"tampered JWT signature                    -> HTTP {r.status_code} {r.json()['detail']['error']}")

        step("3. Fund wallet (sandbox faucet)")
        me = (await c.post("/v1/sandbox/faucet", headers=H, json={"amount_usd": FAUCET_USD})).raise_for_status().json()
        print(f"available={me['available_usd']} escrow={me['escrow_usd']}")

        step(f"4. Order book for {INSTRUMENT}")
        deadline = time.monotonic() + 60
        while True:
            book = (await c.get(f"/v1/book/{INSTRUMENT}")).json()
            if sum(level["tokens"] for level in book["asks"]) >= MIN_BOOK_TOKENS or time.monotonic() > deadline:
                break
            await asyncio.sleep(1)
        print(f"{'side':<5} {'$/1M tokens':>12} {'tokens':>10} {'orders':>7}")
        for level in reversed(book["asks"]):
            print(f"{'ASK':<5} {level['price_usd_per_mtok']:>12} {level['tokens']:>10} {level['orders']:>7}")
        for level in book["bids"]:
            print(f"{'BID':<5} {level['price_usd_per_mtok']:>12} {level['tokens']:>10} {level['orders']:>7}")

        step(f"5. Market BID {BUY_TOKENS} tokens, protection cap ${MAX_PRICE}/1M (escrow before matching)")
        r = await c.post(
            "/v1/orders",
            headers=H,
            json={
                "instrument": INSTRUMENT,
                "side": "bid",
                "order_type": "market",
                "price_usd_per_mtok": MAX_PRICE,
                "quantity_tokens": BUY_TOKENS,
            },
        )
        r.raise_for_status()
        placed = r.json()
        o = placed["order"]
        print(
            f"order {o['order_id']} status={o['status']} filled={o['filled_tokens']}/{o['quantity_tokens']} "
            f"cancelled_remainder={o['cancelled_tokens']}"
        )
        for f in placed["fills"]:
            print(
                f"  FILL {f['tokens_total']:>6} tokens @ ${f['price_usd_per_mtok']}/1M from seller {f['counterparty_id'][:16]}… "
                f"-> allocation {f['trade_id'][:28]}… escrow={f['escrow_usd']}"
            )
        me = (await c.get("/v1/agents/me", headers=H)).json()
        print(f"wallet: available={me['available_usd']} escrow={me['escrow_usd']} (price improvement vs cap already refunded)")

        step(f"6. {JOBS} concurrent inference streams through the proxy router")
        prompts = [f"Explain spot-market inference, variant {i}" for i in range(JOBS)]
        results = await asyncio.gather(*(stream_job(c, H, i + 1, p) for i, p in enumerate(prompts)))

        step("7. Allocations after delivery")
        trades = (await c.get("/v1/trades", headers=H)).json()
        for t in trades:
            print(
                f"  {t['trade_id'][:28]}… seller={t['counterparty_id'][:16]}… ${t['price_usd_per_mtok']}/1M "
                f"used={t['tokens_used']}/{t['tokens_total']} settled={t['settled_usd']} escrow_left={t['escrow_usd']} {t['status']}"
            )
        for t in trades:
            if t["status"] == "active":
                rel = (await c.post(f"/v1/trades/{t['trade_id']}/release", headers=H)).raise_for_status().json()
                print(f"  released {rel['trade_id'][:28]}… -> {rel['status']}, unused escrow refunded")

        step("8. Micro-transaction ledger (buyer, newest first)")
        for e in (await c.get("/v1/ledger?limit=12", headers=H)).json():
            print(f"  {e['kind']:<15} {e['account'].split(':')[-1]:<9} {e['amount_usd']:>15}  {e['memo'] or ''}")
        me = (await c.get("/v1/agents/me", headers=H)).json()
        print(f"final wallet: available={me['available_usd']} escrow={me['escrow_usd']}")

        step("9. Clearinghouse audit (double-entry invariants)")
        audit = (await c.get("/v1/audit")).json()
        print(json.dumps(audit, indent=2))
        ok = audit["ok"] and all(r.get("event") == "done" for r in results)
        print(f"\n{GREEN if ok else RED}{BOLD}SIMULATION {'PASSED' if ok else 'FAILED'}{RESET}")
        return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
