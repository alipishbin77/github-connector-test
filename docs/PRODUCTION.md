# Aether in production: the full picture

## What the platform does

Aether is a **market for AI inference where both sides are software agents**.

- **Buyer agents** need LLM output. They point any OpenAI-compatible client at Aether
  (`base_url=https://<host>/v1`, `api_key=<their key>`) and call `chat.completions` as usual. `model` names an
  instrument such as `llama-3.1-70b-instruct`. For every request Aether buys the cheapest matching capacity on its
  order book, within the agent's price cap. It holds the cost in escrow, streams the answer, and charges only for the
  tokens actually delivered. If a GPU drops mid-answer, the stream continues on another seller from the last
  checkpoint without a gap.
- **Seller agents** have spare GPU capacity. They run `app/seller_gateway.py` next to their own inference server
  (vLLM, SGLang, TGI, llama.cpp), which keeps a price-quoted ask on the book. Sellers are paid per delivered token
  and withdraw to a bank account via Stripe Connect.
- **The platform (you)** keeps a clearing fee on every settlement (`AETHER_FEE_BPS`; the Render blueprint sets 5%).
  Buyers fund balances by card through Stripe Checkout. Fees accumulate in the `house:fees` ledger account, and the
  money itself sits in your Stripe balance.

Nobody ever shares an API key. Buyers hold escrowed *allocations*, the clearinghouse is the only party that calls
sellers, and every seller call carries a single-use JWT bound to the exact request.

### Money flow for one request

```
buyer card ──Stripe Checkout──► platform Stripe balance
                                   │  ledger: house:stripe_deposits → buyer available
buyer available ──bid escrow──► buyer escrow ──per delivered token──► seller available (95%)
                                                                   └► house:fees        (5%)  ← your revenue
seller available ──Stripe Connect transfer──► seller's bank
```

Every movement is a balanced double-entry posting in integer nano-dollars. `GET /v1/audit` (with `X-Admin-Token`)
proves that the ledger sums to zero and that all escrow is backed.

### What an agent sees

```python
from openai import OpenAI
client = OpenAI(base_url="https://api.yourdomain.com/v1", api_key="agt_cli_….aes_…")
r = client.chat.completions.create(
    model="llama-3.1-70b-instruct",
    messages=[{"role": "user", "content": "Summarise this contract…"}],
    max_tokens=400,
    extra_body={"max_price_usd_per_mtok": "0.60"},   # optional price cap
)
print(r.choices[0].message.content, r.usage, r.model_extra["aether"]["cost_usd"])
```

The official `openai` Python SDK has been verified against the server, streaming and non-streaming.

## What is built and verified

| Area | Status |
|---|---|
| Order book, matching, escrow, double-entry ledger, settlement | Done; audited after every CI run |
| OAuth2 JWTs, static API keys, secret rotation, scopes, per-agent rate limits | Done |
| OpenAI-compatible `/v1/chat/completions` (+ streaming, usage), `/v1/models`, `/v1/quote` | Done; official SDK verified |
| Auto-buy per request, multi-seller rollover, checkpoint resume, failover | Done |
| Stripe deposits (Checkout + signed webhook, idempotent) | Done (tested against a mocked Stripe) |
| Seller payouts (Stripe Connect Express onboarding, transfers, reversal on failure) | Done (tested against a mocked Stripe) |
| Seller gateway for vLLM/SGLang/TGI with resume and proprietary-API refusal | Done (tested against a mocked upstream) |
| Self-serve agent signup (per-IP limit), admin revenue stats, crash recovery | Done |
| CI: lint, 32 tests, full Docker end-to-end simulation on Postgres | Done |
| One-click deploy (Render Blueprint) | Done, not yet deployed (needs your Render account) |

## Launch checklist: what only the owner can do

These steps need a legal person, a bank account or a signed contract. Claude cannot do them.

1. **Company and legal.** Form an entity. Publish Terms of Service, a Privacy Policy and an Acceptable Use Policy
   (buyers can generate harmful content through sellers, so you need a rule set and an abuse process). Have counsel
   confirm that the Stripe Connect *platform* model (separate charges and transfers) covers you for holding buyer
   balances and paying sellers where you operate. This is money-transmission territory.
2. **Stripe.** Create the account and activate **Connect (Express)**. Put `sk_live_…` in `AETHER_STRIPE_SECRET_KEY`.
   Add a webhook to `https://<host>/v1/billing/stripe/webhook` for `checkout.session.completed` and
   `checkout.session.async_payment_succeeded`, and put its `whsec_…` in `AETHER_STRIPE_WEBHOOK_SECRET`. Stripe Connect
   also handles seller KYC and tax forms.
3. **Hosting.** In Render: New → Blueprint → this repo. When prompted, fill in the secrets: run
   `python -m app.keygen` for `AETHER_JWT_PRIVATE_KEY_PEM`, set your public URL for `AETHER_JWT_ISSUER` and
   `AETHER_PUBLIC_BASE_URL`, and add the Stripe keys. Attach a custom domain. After that, every merge to `main` redeploys.
4. **Day-one supply (the cold-start problem).** A market with no asks serves no one. The quickest fix is to be the
   first seller yourself: rent a GPU (RunPod, Lambda, Vast), run vLLM with an open-weight model whose license allows
   commercial serving, and run `app/seller_gateway.py` in front of it. Then recruit independent GPU operators.
5. **Demand.** Publish the quickstart above. List the endpoint wherever agent builders look for model providers
   (framework integration docs, MCP/tool directories, model-router listings), and seed a few design partners with
   credit.

## Economics, stated plainly

Fee revenue = settled volume × fee. At open-model prices of about $0.40 per 1M tokens:

| Monthly settled volume | Tokens/month at $0.40/1M | Sustained tokens/sec | Revenue at a 5% fee |
|---|---|---|---|
| $10,000 | 25 B | ~9,600 | $500 |
| $100,000 | 250 B | ~96,000 | $5,000 |
| $1,000,000 | 2.5 T | ~960,000 | $50,000 |

A thin fee on cheap tokens only makes serious money at very large volume. The levers that change this: premium
instruments (large or reasoning models at $2–15/1M), a higher take rate while the market is young (routers typically
charge several percent on credit purchases), and running the house seller at a real spread over your GPU cost.
Measure your own GPU throughput before pricing; the table is arithmetic, not a forecast.

## Engineering roadmap (in priority order)

1. **Tokenizer-based metering.** Bill on tokens counted with the instrument's tokenizer, not on the seller's stream
   events.
2. **Input-token pricing.** Today only completion tokens are billed; prompt tokens are free, which is exploitable.
3. **Alembic migrations** instead of `create_all` (needed before the first schema change after launch).
4. **Observability.** Prometheus metrics, Sentry, alert on audit failures and settlement errors.
5. **Seller trust.** Spot-check prompts to verify the served model, reputation scores, collateral or slashing for
   non-delivery.
6. **Automatic top-ups.** Saved payment method plus off-session charges, so agents never run dry.
7. **MCP server and a thin Python SDK**, so agents can discover and use the market as tools.
8. **Horizontal scale.** Multiple clearinghouse replicas (the matching engine is already multi-writer safe) and Redis
   high availability.

## Operating it

- `GET /v1/admin/stats` (`X-Admin-Token`): revenue, settled volume, deposits, payouts, agents, jobs in the last 24 hours.
- `GET /v1/audit` (`X-Admin-Token`): ledger invariants. Alert if `ok` is ever false.
- Logs: `FILL`, `ROUTE`, `SETTLED`, `DEPOSIT`, `PAYOUT` lines trace every dollar.
