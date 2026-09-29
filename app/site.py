"""Public front door: a landing page for people and an llms.txt quickstart for
agents. Both are generated from live settings, so the payment instructions
always match what the server actually accepts."""

from html import escape

from fastapi import APIRouter
from fastapi.responses import HTMLResponse, PlainTextResponse

from .config import settings
from .crypto_payments import configured_networks

router = APIRouter(include_in_schema=False)


def _payments_text() -> str:
    networks = configured_networks()
    if networks:
        symbols = sorted({n.token_symbol for n in networks})
        names = ", ".join(n.name for n in networks)
        return (
            f"{'/'.join(symbols)} on {names} to the treasury {settings.crypto_treasury_address} "
            "(same address on every network), sent from a wallet you have linked"
        )
    if settings.stripe_secret_key:
        return "card top-ups via Stripe Checkout (POST /v1/billing/deposits)"
    return "not enabled yet on this clearinghouse"


def _fee_pct() -> str:
    return f"{settings.fee_bps / 100:g}%"


LLMS_TXT = """\
# Aether — inference market for AI agents

> Buy LLM inference from independent GPU sellers through one OpenAI-compatible API.
> Pay only for tokens delivered. Streams survive seller failures (checkpoint + failover).

Base URL: {base}/v1

## Buy inference (agents)
1. Register: POST {base}/v1/agents  {{"name": "<agent name>", "scopes": ["buy_inference"]}}
   -> returns api_key (shown once; store it).
2. Fund your balance with {payments}.
   - Link your wallet: GET {base}/v1/billing/crypto/link-challenge?address=<0x...>
     sign the returned "message" with that wallet (EIP-191 personal_sign), then
     POST {base}/v1/billing/crypto/wallets {{"address": ..., "nonce": ..., "signature": ...}}
   - Send native USDC from that wallet on any listed network (cheapest: Base, Arbitrum, OP Mainnet, Polygon).
     It is credited after that network's confirmations. Bridged USDC.e and other tokens are not credited.
   - Check: GET {base}/v1/agents/me  (Authorization: Bearer <api_key>)
3. Call any OpenAI-style client with base_url={base}/v1 and api_key=<api_key>:
   POST {base}/v1/chat/completions {{"model": "<instrument>", "messages": [...], "max_tokens": 256,
     "max_price_usd_per_mtok": "0.60"}}
   Available models and live prices: GET {base}/v1/models . Price a job first: GET {base}/v1/quote/<model>?tokens=1000

## Sell inference (GPU owners)
Register with scope "sell_compute", then run app/seller_gateway.py next to your own vLLM/SGLang/TGI server.
It keeps your capacity on the order book and gets paid per delivered token (minus a {fee} clearing fee).
Withdraw: POST {base}/v1/billing/crypto/withdrawals {{"amount_usd": "25.00", "to_address": "<your linked wallet>"}}

## Feedback (please!)
Anything broken, missing or confusing: POST {base}/v1/feedback {{"category": "bug|feature|pricing|docs|other", "message": "..."}}
No account needed. Include your api_key as Bearer to link it to your agent.

## Reference
Agent card: {base}/.well-known/agent.json   OpenAPI: {base}/openapi.json   Interactive docs: {base}/docs
Every 402 error includes "how_to_fund" with the exact payment steps.
"""


@router.get("/llms.txt", response_class=PlainTextResponse)
async def llms_txt():
    return LLMS_TXT.format(
        base=settings.public_base_url.rstrip("/"),
        payments=_payments_text(),
        fee=_fee_pct(),
    )


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Aether — inference market for AI agents</title>
<meta name="description" content="Agents buy LLM inference from GPU sellers through one OpenAI-compatible API and pay per delivered token.">
<style>
:root{{--bg:#fbfbfa;--fg:#1d1d1b;--mute:#6b6a66;--line:#e4e2dc;--card:#fff;--accent:#2f5bd3}}
@media (prefers-color-scheme:dark){{:root{{--bg:#141413;--fg:#ecebe7;--mute:#9d9b94;--line:#2c2b28;--card:#1c1c1a;--accent:#8aa8ff}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--fg);font:16px/1.55 ui-sans-serif,system-ui,-apple-system,Segoe UI,sans-serif}}
main{{max-width:880px;margin:0 auto;padding:48px 16px 64px}}h1{{font-size:clamp(28px,5vw,40px);line-height:1.15;margin:0 0 12px}}
h2{{font-size:20px;margin:40px 0 12px}}p.lead{{color:var(--mute);font-size:18px;margin:0 0 24px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px}}
.card b{{display:block;margin-bottom:4px}}.mute{{color:var(--mute)}}
pre{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;overflow-x:auto;font-size:13.5px}}
code{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}}table{{width:100%;border-collapse:collapse;font-size:15px}}
th,td{{text-align:left;padding:8px 6px;border-bottom:1px solid var(--line)}}th{{color:var(--mute);font-weight:500}}
td.num{{text-align:right;font-variant-numeric:tabular-nums}}a{{color:var(--accent)}}
</style></head><body><main>
<h1>Aether</h1>
<p class="lead">A spot market where AI agents buy LLM inference from independent GPU sellers through one
OpenAI-compatible API, and pay only for the tokens they receive.</p>
<div class="grid">
<div class="card"><b>Drop-in API</b><span class="mute">Point any OpenAI client at <code>{base}/v1</code>. The model name selects the market.</span></div>
<div class="card"><b>Pay per token</b><span class="mute">Funds are escrowed and settled per delivered token. Unused capacity is refunded.</span></div>
<div class="card"><b>Survives failures</b><span class="mute">If a seller drops mid-answer, the stream resumes on another seller from the last checkpoint.</span></div>
</div>

<h2>Live market</h2>
<table><thead><tr><th>Model</th><th class="num">Best price / 1M tokens</th><th class="num">Tokens on offer</th></tr></thead>
<tbody id="book"><tr><td colspan="3" class="mute">Loading…</td></tr></tbody></table>

<h2>Quickstart for agents</h2>
<pre><code>curl -s {base}/v1/agents -H 'content-type: application/json' \\
  -d '{{"name":"my-agent","scopes":["buy_inference"]}}'      # returns api_key (shown once)

from openai import OpenAI
client = OpenAI(base_url="{base}/v1", api_key="&lt;api_key&gt;")
client.chat.completions.create(model="&lt;model&gt;", max_tokens=256,
    messages=[{{"role":"user","content":"hello"}}],
    extra_body={{"max_price_usd_per_mtok":"0.60"}})</code></pre>
<p><b>Paying:</b> {payments}. Full steps for agents are in <a href="/llms.txt">/llms.txt</a>; every endpoint is in <a href="/docs">/docs</a>.</p>

<h2>For machines</h2>
<p>Discovery: <a href="/.well-known/agent.json">/.well-known/agent.json</a> · <a href="/llms.txt">/llms.txt</a> · <a href="/openapi.json">/openapi.json</a>.
Every <code>402</code> response carries <code>how_to_fund</code> with exact payment steps. Tell us what to fix:
<code>POST /v1/feedback</code> (no account needed).</p>

<h2>For GPU owners</h2>
<p>Run the seller gateway next to your own vLLM, SGLang or TGI server. It lists your capacity on the order book and you are paid
per delivered token, less a {fee} clearing fee, withdrawable to your own wallet.</p>
</main>
<script>
fetch('/v1/models').then(r=>r.json()).then(d=>{{
  const rows=(d.data||[]).map(m=>`<tr><td><code>${{m.id}}</code></td><td class="num">${{m.best_ask_usd_per_mtok?('$'+m.best_ask_usd_per_mtok):'—'}}</td><td class="num">${{(m.ask_depth_tokens||0).toLocaleString()}}</td></tr>`);
  document.getElementById('book').innerHTML=rows.join('')||'<tr><td colspan="3" class="mute">No sellers listed yet.</td></tr>';
}}).catch(()=>{{document.getElementById('book').innerHTML='<tr><td colspan="3" class="mute">Market data unavailable.</td></tr>'}});
</script></body></html>
"""


@router.get("/", response_class=HTMLResponse)
async def landing():
    return PAGE.format(
        base=escape(settings.public_base_url.rstrip("/")),
        payments=escape(_payments_text()),
        fee=escape(_fee_pct()),
    )
