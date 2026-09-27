# Project Aether — M2M Inference Clearinghouse (MVP)

AI agents buy and sell **inference units** (tokens of a named model, e.g. `llama-3.1-70b-instruct`) on a continuous
double-auction spot market. **Credentials never change hands.** Buyers hold escrowed *allocations*, and the clearinghouse
proxies every request to the seller's own stateless endpoint using a single-use, body-bound delivery JWT. Sellers are paid
in nano-dollars for exactly the tokens the proxy streamed.

**Going to production:** see [docs/PRODUCTION.md](docs/PRODUCTION.md) for what the live platform does, how money
moves, the launch checklist, and the roadmap.

**Agents can connect with any OpenAI-compatible client:**

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="<client_id>.<client_secret>")  # api_key from POST /v1/agents
client.chat.completions.create(
    model="llama-3.1-70b-instruct",
    messages=[{"role": "user", "content": "hi"}],
    max_tokens=64,
    extra_body={"max_price_usd_per_mtok": "0.50"},
)
```

Each call market-buys any missing capacity within the price cap, escrows it, streams the answer, and settles per
delivered token.

> **Scope note.** Abstracting keys behind a proxy does **not** by itself make it permissible to resell a proprietary API
> (OpenAI, Anthropic, Google…). If a seller's endpoint just forwards to their own vendor key, that is still resale of
> access, which those providers' terms generally prohibit. The model this MVP is built for is sellers running models they
> have the rights to serve (open-weight models on their own or rented GPUs), or providers who explicitly allow resale.

## Architecture

```
            OAuth2 client_credentials                    ┌──────────────────────────────┐
 Buyer ──── POST /oauth/token ──► JWT (buy_inference) ──►│  FastAPI clearinghouse       │
 agent      POST /v1/orders  (bid, escrowed first) ─────►│  auth.py      RS256 JWT/JWKS │
            POST /v1/inference (SSE) ◄──── stream ──────│  exchange.py  clearing+escrow│
                                                         │  proxy_router.py (asyncio)   │
 Seller ─── POST /v1/orders (ask) ──────────────────────►│                              │
 agent  ◄── POST /v1/generate + delivery JWT ───────────│                              │
   (verifies via /.well-known/jwks.json)                 └──────┬──────────────┬────────┘
                                                                │              │
                          Redis: Lua matching engine (ZSET book),│              │ PostgreSQL: agents, orders,
                          match Stream (durable fill log),        │              │ trades/allocations, double-entry
                          Pub/Sub market data, job checkpoints   ▼              ▼ ledger (source of truth)
```

**Order → allocation → delivery → settlement**

1. **Escrow before matching.** A bid locks `price_cap × qty` from `available` into `escrow` before it reaches the book.
2. **Match.** One Lua script matches atomically by price, then time (FIFO), with self-trade prevention, and appends the
   result to a Redis Stream. Fills execute at the maker's price, and price improvement is refunded at once.
3. **Apply exactly once.** Each stream event is applied to PostgreSQL once, guarded by `applied_match_events`. The request
   applies its own event inline; a consumer-group worker applies anything left over after a crash. PostgreSQL is
   authoritative, and on startup the Redis book is rebuilt from it.
4. **Route.** `/v1/inference` reserves `max_tokens` on the cheapest escrowed allocation (row-locked, so concurrent
   requests can't overdraw it) or returns **402**. It then mints a 60-second delivery JWT
   (`aud=aether-seller:<id>`, `jti` single use, `body_sha256` binding) and streams the seller's SSE response back.
5. **Checkpoint / retry.** Delivered tokens are batched into a Redis Stream. If the seller drops the connection, times
   out, returns 5xx, or ends without a `done` marker (spot preemption), the proxy resumes on the same allocation with
   `resume_from` and `prefix`. After `max_attempts_per_allocation` it fails over to another escrowed allocation. The
   buyer sees one continuous, gap-free stream.
6. **Settle.** Buyer escrow moves to seller `available` plus the `house:fees` account for exactly the delivered tokens.
   The rest of the reservation is released. This runs even if the buyer disconnects. If the proxy process dies, a sweep
   settles orphaned jobs from the checkpoint once their heartbeat lapses.

**Proxy choice: Python asyncio.** The router is I/O-bound: it relays chunks and runs a few short transactions per request.
One event loop with a pooled `httpx.AsyncClient` multiplexes thousands of streams and shares the ledger code with no RPC hop.
A Go data plane becomes worthwhile once per-chunk CPU work (tokenizer-based metering) dominates.

## Data model (`app/models.py`)

All money is **integer nano-USD** (1e-9). Prices are **integer nano-USD per token**: $0.35 per 1M tokens = 350. That puts
the tick at $0.001/1M, and every cost is an exact integer product with no rounding. Fees are computed cumulatively per
allocation (`floor(gross_total × bps / 10⁴)`), so they never drift across many micro-settlements.

| Table | Purpose / key columns |
|---|---|
| `agents` | `client_id`, `client_secret_hash` (scrypt), `allowed_scopes`, `endpoint_url`, `balance_available_nanos`, `balance_escrow_nanos`, `token_version` (bump to revoke all JWTs) |
| `orders` | `side` bid/ask, `order_type` limit/market, `time_in_force` gtc/ioc, `price_npt`, `quantity/filled/cancelled_tokens`, `escrow_nanos` (bids: `price × open`), `seq` (time priority), `expires_at` |
| `trades` | A fill, and for the buyer an **allocation**: `tokens_total/used/reserved`, `escrow_nanos` (= `price × unused`), `gross_settled_nanos`, `fee_nanos`, `status` active/exhausted/released/expired |
| `trade_ledger` | Append-only **double-entry** journal: every `tx_id` sums to 0; accounts `agent:<id>:available`, `agent:<id>:escrow`, `house:fees`, `house:sandbox_mint` |
| `applied_match_events` | Exactly-once guard for match-stream events |
| `inference_jobs` | Per-request status, tokens, cost, attempts, failovers, per-seller segments |

`GET /v1/audit` (sandbox) checks the invariants: the ledger sums to zero, every transaction balances, cached balances
equal the ledger, and each agent's escrow equals its open-bid escrow plus its allocation escrow.

## JWT payloads (RS256, `kid` = RFC 7638 thumbprint, public keys at `/.well-known/jwks.json`)

Access token (from `POST /oauth/token`, `grant_type=client_credentials`, HTTP Basic or form auth):

```json
{
  "iss": "http://clearinghouse:8000", "sub": "agt_7bf7…", "aud": "aether-clearinghouse",
  "iat": 1790508813, "nbf": 1790508813, "exp": 1790509713, "jti": "0dbd3f97…",
  "client_id": "agt_cli_f17d…", "scope": "buy_inference", "tv": 0, "typ": "access"
}
```

Scopes: `buy_inference` (bid, consume, release allocations) and `sell_compute` (ask, register endpoint). A token can be
revoked by `jti` (`POST /oauth/revoke`), or all of an agent's tokens at once via `tv`.

Delivery token (proxy → seller, one per seller call):

```json
{
  "iss": "…", "sub": "job_0d41…", "aud": "aether-seller:agt_2edb…", "exp": "iat + 60",
  "jti": "…", "typ": "delivery", "trade_id": "trd_…", "instrument": "llama-3.1-70b-instruct",
  "max_tokens": 150, "resume_from": 25, "body_sha256": "<sha256 of the exact request body>"
}
```

## Files

| Path | What it is |
|---|---|
| `app/main.py` | FastAPI app, CORS, Redis pool, httpx pool, startup recovery (drain stream → reconcile book), background workers |
| `app/auth.py` | OAuth2 client-credentials grant, scrypt secrets, JWT issue/verify, scope dependencies, revocation, JWKS, delivery tokens |
| `app/matching_engine.py` | Redis Lua continuous double auction (price/time FIFO, self-trade prevention, expiry, durable match stream) |
| `app/exchange.py` | Order entry, exactly-once fill application, escrow moves, allocation reservation, job settlement, stream worker, reconcile |
| `app/proxy_router.py` | Escrow gate, delivery-token minting, async SSE relay, checkpoint/resume/failover, settlement, orphan recovery |
| `app/ledger.py` | Double-entry postings with row-locked balance checks |
| `app/api.py` | Agent registry, faucet, orders, book, market-data SSE, allocations, ledger, audit |
| `app/openai_compat.py` | OpenAI-compatible `/v1/chat/completions` (streaming + usage), `/v1/models`, `/v1/quote`, auto-buy |
| `app/payments.py` | Stripe Checkout deposits (signed, idempotent webhook), Stripe Connect onboarding and payouts |
| `app/seller_gateway.py` | **Production seller**: fronts your vLLM/SGLang/TGI server, keeps an ask listed, verifies delivery JWTs, resumes from prefix |
| `app/keygen.py` | Generates the RS256 key for `AETHER_JWT_PRIVATE_KEY_PEM` |
| `render.yaml`, `.github/workflows/ci.yml` | One-click Render deploy (app, Postgres, Key Value); CI running lint, tests and Docker end-to-end |
| `app/seller_agent_sim.py` | Mock seller: self-registers, lists capacity, verifies delivery JWTs, serves SSE, simulates spot preemption |
| `app/buyer_agent_sim.py` | Mock buyer: the full M2M negotiation end to end |
| `Dockerfile`, `docker-compose.yml` | App image, plus Postgres 16, Redis 7 (AOF), clearinghouse, `seller-a` (cheap, flaky), `seller-b` (reliable), `buyer` |
| `tests/` | 32 async tests: matching, escrow, auth, API keys, rate limits, resume, failover, rollover, overdraw races, orphan recovery, OpenAI format, Stripe deposits and payouts, seller gateway |

## Running it

```bash
docker compose up -d --build --wait      # postgres, redis, clearinghouse, seller-a, seller-b
docker compose run --rm --no-deps buyer  # the buyer agent's end-to-end simulation
docker compose logs -f clearinghouse seller-a
open http://localhost:8000/docs          # OpenAPI UI
docker compose down -v                   # tear down, wiping volumes
```

Local development and tests (need `redis-server` on PATH; the tests use SQLite plus a throwaway Redis):

```bash
python -m venv .venv && . .venv/bin/activate && pip install -r requirements-dev.txt
pytest -q
```

### Expected output (from an actual `docker compose run --rm --no-deps buyer`, abridged)

```
== 2. The M2M boundary holds =============================================
sell attempt with a buy_inference token   -> HTTP 403 insufficient_scope (Bearer realm="aether", error="insufficient_scope", scope="sell_compute")
inference before escrowing any funds      -> HTTP 402 no_escrowed_allocation
tampered JWT signature                    -> HTTP 401 invalid_token

== 4. Order book for llama-3.1-70b-instruct ==============================
side   $/1M tokens     tokens  orders
ASK            0.4       5000       1
ASK           0.35        400       1

== 5. Market BID 1000 tokens, protection cap $0.50/1M (escrow before matching)
order ord_6645… status=filled filled=1000/1000 cancelled_remainder=0
  FILL    400 tokens @ $0.35/1M from seller agt_9b4c3e5d7982… -> allocation trd_1790507830419_0_0… escrow=$0.000140000
  FILL    600 tokens @ $0.4/1M from seller agt_b242131598ab… -> allocation trd_1790507830419_0_1… escrow=$0.000240000
wallet: available=$0.999620000 escrow=$0.000380000 (price improvement vs cap already refunded)

== 6. 3 concurrent inference streams through the proxy router ============
[job1] routed job=job_83cb… -> seller=agt_9b4c3e5d7982… trade=trd_1790507830419_0_0… @ $0.35/1M reserved=150
[job3] routed job=job_1d8c… -> seller=agt_b242131598ab… trade=trd_1790507830419_0_1… @ $0.4/1M reserved=150
[job1] interrupted: stream ended without completion marker (instance preempted)
[job1] RESUME on seller=agt_9b4c3e5d7982… from checkpoint token #25
[job1] interrupted: stream ended without completion marker (instance preempted)
[job1] FAILOVER to seller=agt_b242131598ab… from checkpoint token #50
[job1] done: 150 tokens, cost $0.000057500, attempts=3, failovers=1, finish=length
[job1]   segment seller=agt_9b4c3e5d7982… trade=trd_1790507830419_0_0… tokens=50
[job1]   segment seller=agt_b242131598ab… trade=trd_1790507830419_0_1… tokens=100

== 8. Micro-transaction ledger (buyer, newest first) =====================
  escrow_release  available    $0.000105000  allocation released by buyer
  settlement      escrow      $-0.000040000  job_798b…: 100 tokens
  settlement      escrow      $-0.000017500  job_798b…: 50 tokens
  escrow_release  available    $0.000060000  price improvement
final wallet: available=$0.999825000 escrow=$0.000000000

== 9. Clearinghouse audit (double-entry invariants) ======================
{ "ledger_sum_nanos": 0, "unbalanced_transactions": [], "cached_balance_mismatches": [],
  "escrow_backing_mismatches": [], "house_fees_usd": "$0.000001750", "ok": true }

SIMULATION PASSED
```

Clearinghouse log for the same run:

```
aether.exchange: FILL llama-3.1-70b-instruct 400 tokens @ $0.35/1M buyer=agt_7bf7… seller=agt_2edb… (taker=bid)
aether.proxy: ROUTE job=job_0d41… buyer=agt_7bf7… -> seller=agt_2edb… trade=trd_…_0_0 reserved=150
aether.proxy: job=job_0d41… seller=agt_2edb… interrupted at token 25 (attempt 1): stream ended without completion marker (instance preempted)
aether.proxy: job=job_0d41… seller=agt_2edb… interrupted at token 50 (attempt 2): stream ended without completion marker (instance preempted)
aether.proxy: SETTLED job=job_0d41… tokens=150 cost=$0.000057500 segments=[('trd_…_0_0', 50), ('trd_…_0_1', 100)] status=completed
```

## API summary

| Method & path | Scope | Notes |
|---|---|---|
| `POST /v1/agents` | — (sandbox) / `X-Admin-Token` | Returns `client_id` and `client_secret` (the secret is shown once) |
| `POST /oauth/token`, `POST /oauth/revoke` | client creds / bearer | RFC 6749 §4.4, RFC 7009 subset |
| `PUT /v1/agents/me/endpoint` | `sell_compute` | Seller webhook. `AETHER_ALLOW_PRIVATE_SELLER_URLS=false` enforces https and public IPs |
| `POST /v1/orders`, `GET/DELETE /v1/orders/{id}` | bid→`buy_inference`, ask→`sell_compute` | `price_usd_per_mtok`, `quantity_tokens`, `order_type`, `time_in_force`, `ttl_seconds` |
| `GET /v1/book/{instrument}`, `GET /v1/market/{instrument}/stream` | public | Depth; trade prints over SSE (Redis Pub/Sub) |
| `POST /v1/inference` | `buy_inference` | SSE events: `meta`, token `data`, `resume`, `done` / `error` |
| `GET /v1/jobs/{id}` | `buy_inference` | Status plus checkpointed text (recovers output if the buyer's connection dropped) |
| `GET /v1/trades`, `POST /v1/trades/{id}/release` | any / `buy_inference` | Allocations; release refunds unused escrow |
| `POST /v1/chat/completions`, `GET /v1/models`, `GET /v1/quote/{model}` | `buy_inference` / public | OpenAI-compatible; `max_price_usd_per_mtok` and `auto_buy` extensions |
| `POST /v1/agents/me/rotate-secret` | any | New secret; old secret, API key and JWTs stop working |
| `POST /v1/billing/deposits`, `POST /v1/billing/stripe/webhook` | any / Stripe-signed | Card top-ups via Stripe Checkout |
| `POST /v1/billing/connect/onboard`, `POST/GET /v1/billing/withdrawals` | `sell_compute` | Stripe Connect KYC and payouts |
| `GET /v1/ledger` | any | The agent's ledger entries |
| `GET /v1/audit`, `GET /v1/admin/stats` | sandbox or `X-Admin-Token` | Invariants; revenue, volume and activity |
| `POST /v1/sandbox/faucet` | sandbox only | Test money |

## Known gaps before real money

The full list, with priorities, is in [docs/PRODUCTION.md](docs/PRODUCTION.md#engineering-roadmap-in-priority-order).
The most important:

- **Metering trust.** The proxy counts SSE token events, so a dishonest seller could split output into more "tokens".
  Billing should use the instrument's tokenizer.
- **Prompt tokens are not billed yet**, and nothing yet verifies which model a seller actually runs.
- **Operations.** The schema is created with `create_all` (Alembic migrations needed before the first schema change),
  and the Lua scripts assume a single Redis primary.
- **Regulation.** Real money flows through Stripe Connect as the platform, but holding balances and paying sellers
  still needs a legal review in your jurisdiction.
