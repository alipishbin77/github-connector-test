import json

import httpx
import pytest
from eth_account import Account
from eth_account.messages import encode_defunct

from app import crypto_payments as cp
from app.config import settings

from .helpers import audit_ok, me, new_agent

TREASURY = "0x" + "7e" * 20
TOKEN = "0x" + "a0" * 20
TOKEN_B = "0x" + "b0" * 20
ADMIN = "test-admin-token-0123456789abcdef"


def topic(addr: str) -> str:
    return "0x" + "0" * 24 + addr[2:].lower()


class FakeChain:
    """In-memory EVM JSON-RPC node: blocks, Transfer logs, receipts, balanceOf, getCode."""

    def __init__(self, token: str = TOKEN):
        self.token = token
        self.head = 1_000
        self.logs: list[dict] = []
        self.receipts: dict[str, dict] = {}
        self.balance_units = 0

    def transfer_in(self, sender: str, units: int, block: int, tx: str, index: int = 0) -> None:
        self.logs.append(
            {
                "address": self.token,
                "topics": [cp.TRANSFER_TOPIC, topic(sender), topic(TREASURY)],
                "data": hex(units),
                "blockNumber": hex(block),
                "transactionHash": tx,
                "logIndex": hex(index),
                "removed": False,
            }
        )
        self.balance_units += units

    def transfer_out(self, to: str, units: int, block: int, tx: str, status: str = "0x1") -> None:
        self.receipts[tx] = {
            "status": status,
            "blockNumber": hex(block),
            "logs": [{"address": self.token, "topics": [cp.TRANSFER_TOPIC, topic(TREASURY), topic(to)], "data": hex(units)}],
        }
        self.balance_units -= units

    async def handle(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        method, params = body["method"], body["params"]
        if method == "eth_blockNumber":
            result = hex(self.head)
        elif method == "eth_getLogs":
            f = params[0]
            lo, hi = int(f["fromBlock"], 16), int(f["toBlock"], 16)
            result = [
                entry for entry in self.logs if lo <= int(entry["blockNumber"], 16) <= hi and entry["topics"][2] == f["topics"][2]
            ]
        elif method == "eth_getTransactionReceipt":
            result = self.receipts.get(params[0])
        elif method == "eth_call":
            result = hex(self.balance_units)
        elif method == "eth_getCode":
            result = "0x"
        else:
            return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "error": {"message": "unsupported"}})
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": result})


_next_chain_id = iter(range(900_000, 999_999, 2))


@pytest.fixture
async def chain(app, monkeypatch):
    """Two networks sharing one treasury: `chain` (default for payouts) and `chain.other`."""
    cid = next(_next_chain_id)  # fresh chain ids => fresh deposit cursors per test
    fake_a, fake_b = FakeChain(TOKEN), FakeChain(TOKEN_B)
    net_a = cp.Network("net-a", "Net A", cid, "http://rpc-a.test", TOKEN, confirmations=5)
    net_b = cp.Network("net-b", "Net B", cid + 1, "http://rpc-b.test", TOKEN_B, confirmations=5)
    monkeypatch.setattr(settings, "crypto_treasury_address", TREASURY)
    monkeypatch.setattr(settings, "admin_token", ADMIN)
    rails = {}
    for net, fake in ((net_a, fake_a), (net_b, fake_b)):
        client = cp.EvmClient(httpx.AsyncClient(transport=httpx.MockTransport(fake.handle)), net.rpc_url)
        fake.watcher = cp.DepositWatcher(net, client)
        rails[net.chain_id] = cp.Rail(net, client, fake.watcher)
    monkeypatch.setattr(app.state, "crypto", cp.CryptoRails(rails), raising=False)
    fake_a.other = fake_b
    yield fake_a
    for rail in rails.values():
        await rail.client.aclose()


async def link(client, agent, account) -> httpx.Response:
    challenge = (
        await client.get(f"/v1/billing/crypto/link-challenge?address={account.address}", headers=agent["headers"])
    ).json()
    sig = account.sign_message(encode_defunct(text=challenge["message"])).signature.hex()
    return await client.post(
        "/v1/billing/crypto/wallets",
        headers=agent["headers"],
        json={"address": account.address, "nonce": challenge["nonce"], "signature": "0x" + sig.removeprefix("0x")},
    )


def tx(n: int) -> str:
    return "0x" + f"{n:064x}"


async def test_linked_deposit_is_credited_once_after_confirmations(client, chain):
    agent = await new_agent(client, "payer", ["buy_inference"])
    wallet = Account.create()
    assert (await link(client, agent, wallet)).status_code == 201
    info = (await client.get("/v1/billing/crypto", headers=agent["headers"])).json()
    assert info["treasury_address"] == TREASURY and info["linked_wallets"] == [wallet.address.lower()]
    assert [n["network"] for n in info["networks"]] == ["net-a", "net-b"]

    await chain.watcher.poll()  # initialise the cursor at the current safe head
    chain.transfer_in(wallet.address, 25_000_000, block=chain.head + 1, tx=tx(1))  # 25 USDC
    chain.head += 2
    await chain.watcher.poll()
    assert (await me(client, agent))["available_nanos"] == 0  # not enough confirmations yet
    chain.head += 10
    await chain.watcher.poll()
    await chain.watcher.poll()  # re-scan / replay must not double-credit
    assert (await me(client, agent))["available_nanos"] == 25 * 10**9
    deposits = (await client.get("/v1/billing/crypto/deposits", headers=agent["headers"])).json()
    assert [d["status"] for d in deposits] == ["credited"] and deposits[0]["amount"] == "25 USDC on Net A"
    await audit_ok(client)


async def test_unknown_sender_is_held_until_that_wallet_links(client, chain):
    agent = await new_agent(client, "late", ["buy_inference"])
    wallet = Account.create()
    await chain.watcher.poll()
    chain.transfer_in(wallet.address, 7_500_000, block=chain.head + 1, tx=tx(2))
    chain.head += 20
    await chain.watcher.poll()
    assert (await me(client, agent))["available_nanos"] == 0
    r = await link(client, agent, wallet)
    assert r.json()["credited_now_usd"] == "$7.500000000"
    assert (await me(client, agent))["available_nanos"] == 7_500_000_000
    await audit_ok(client)


async def test_wallet_link_requires_a_valid_signature_from_that_address(client, chain):
    agent = await new_agent(client, "a", ["buy_inference"])
    other = await new_agent(client, "b", ["buy_inference"])
    wallet, impostor = Account.create(), Account.create()
    challenge = (await client.get(f"/v1/billing/crypto/link-challenge?address={wallet.address}", headers=agent["headers"])).json()
    forged = impostor.sign_message(encode_defunct(text=challenge["message"])).signature.hex()
    r = await client.post(
        "/v1/billing/crypto/wallets",
        headers=agent["headers"],
        json={"address": wallet.address, "nonce": challenge["nonce"], "signature": "0x" + forged.removeprefix("0x")},
    )
    assert r.status_code == 400
    # the challenge is single-use even after a failed attempt
    good = wallet.sign_message(encode_defunct(text=challenge["message"])).signature.hex()
    r = await client.post(
        "/v1/billing/crypto/wallets",
        headers=agent["headers"],
        json={"address": wallet.address, "nonce": challenge["nonce"], "signature": "0x" + good.removeprefix("0x")},
    )
    assert r.status_code == 400
    assert (await link(client, agent, wallet)).status_code == 201
    assert (await link(client, other, wallet)).status_code == 409  # can't steal another agent's wallet


async def test_payout_is_debited_then_paid_only_after_onchain_verification(client, chain):
    seller = await new_agent(client, "seller", ["sell_compute"])
    wallet, stranger = Account.create(), Account.create()
    await link(client, seller, wallet)
    await chain.watcher.poll()
    chain.transfer_in(wallet.address, 50_000_000, block=chain.head + 1, tx=tx(3))  # seed balance
    chain.head += 20
    await chain.watcher.poll()

    r = await client.post(
        "/v1/billing/crypto/withdrawals", headers=seller["headers"], json={"amount_usd": "20.00", "to_address": stranger.address}
    )
    assert r.status_code == 409  # only to its own linked wallet
    r = await client.post(
        "/v1/billing/crypto/withdrawals", headers=seller["headers"], json={"amount_usd": "20.00", "to_address": wallet.address}
    )
    payout = r.json()
    assert r.status_code == 201 and payout["status"] == "pending"
    assert (await me(client, seller))["available_nanos"] == 30 * 10**9

    admin = {"X-Admin-Token": ADMIN}
    assert (await client.get("/v1/admin/crypto/payouts", headers={"X-Admin-Token": "wrong"})).status_code == 403
    queue = (await client.get("/v1/admin/crypto/payouts", headers=admin)).json()["payouts"]
    assert [p["payout_id"] for p in queue] == [payout["payout_id"]] and queue[0]["amount_units"] == 20_000_000

    url = f"/v1/admin/crypto/payouts/{payout['payout_id']}/paid"
    chain.transfer_out(wallet.address, 19_000_000, block=chain.head, tx=tx(10))  # wrong amount
    chain.head += 20
    assert (await client.post(url, headers=admin, json={"tx_hash": tx(10)})).status_code == 409
    chain.transfer_out(wallet.address, 20_000_000, block=chain.head, tx=tx(11))  # right, but too fresh
    assert (await client.post(url, headers=admin, json={"tx_hash": tx(11)})).status_code == 409
    chain.head += 20
    r = await client.post(url, headers=admin, json={"tx_hash": tx(11)})
    assert r.status_code == 200 and r.json()["status"] == "paid"
    assert (await client.post(url, headers=admin, json={"tx_hash": tx(11)})).status_code == 409

    solvency = (await client.get("/v1/admin/crypto/solvency", headers=admin)).json()
    assert solvency["pending_payouts_usd"] == "$0.000000000" and solvency["paid_out_total_usd"] == "$20.000000000"
    await audit_ok(client)


async def test_cancelled_payout_refunds_the_agent(client, chain):
    seller = await new_agent(client, "s", ["sell_compute"])
    wallet = Account.create()
    await link(client, seller, wallet)
    await chain.watcher.poll()
    chain.transfer_in(wallet.address, 15_000_000, block=chain.head + 1, tx=tx(4))
    chain.head += 20
    await chain.watcher.poll()
    r = await client.post(
        "/v1/billing/crypto/withdrawals", headers=seller["headers"], json={"amount_usd": "15.00", "to_address": wallet.address}
    )
    assert (await me(client, seller))["available_nanos"] == 0
    r = await client.post(f"/v1/admin/crypto/payouts/{r.json()['payout_id']}/cancel", headers={"X-Admin-Token": ADMIN})
    assert r.json()["status"] == "cancelled"
    assert (await me(client, seller))["available_nanos"] == 15 * 10**9
    await audit_ok(client)


async def test_crypto_disabled_without_treasury(app, client, monkeypatch):
    monkeypatch.setattr(app.state, "crypto", None, raising=False)
    agent = await new_agent(client, "x", ["buy_inference"])
    assert (await client.get("/v1/billing/crypto", headers=agent["headers"])).status_code == 503


async def test_deposits_and_payouts_on_a_second_network(client, chain):
    agent = await new_agent(client, "multi", ["sell_compute"])
    wallet = Account.create()
    await link(client, agent, wallet)
    other = chain.other
    await other.watcher.poll()
    other.transfer_in(wallet.address, 30_000_000, block=other.head + 1, tx=tx(20))
    other.head += 20
    await other.watcher.poll()
    assert (await me(client, agent))["available_nanos"] == 30 * 10**9

    r = await client.post(
        "/v1/billing/crypto/withdrawals",
        headers=agent["headers"],
        json={"amount_usd": "12.00", "to_address": wallet.address, "network": "net-b"},
    )
    payout = r.json()
    assert r.status_code == 201 and payout["network"] == "net-b"
    bad = await client.post(
        "/v1/billing/crypto/withdrawals",
        headers=agent["headers"],
        json={"amount_usd": "12.00", "to_address": wallet.address, "network": "dogechain"},
    )
    assert bad.status_code == 422

    admin = {"X-Admin-Token": ADMIN}
    url = f"/v1/admin/crypto/payouts/{payout['payout_id']}/paid"
    chain.transfer_out(wallet.address, 12_000_000, block=chain.head, tx=tx(21))  # sent on the WRONG network
    chain.head += 20
    r = await client.post(url, headers=admin, json={"tx_hash": tx(21)})
    assert r.status_code == 409 and "not found on Net B" in r.json()["detail"]
    other.transfer_out(wallet.address, 12_000_000, block=other.head, tx=tx(22))
    other.head += 20
    assert (await client.post(url, headers=admin, json={"tx_hash": tx(22)})).json()["status"] == "paid"

    solvency = (await client.get("/v1/admin/crypto/solvency", headers=admin)).json()
    assert [n["network"] for n in solvency["networks"]] == ["net-a", "net-b"]
    assert solvency["networks"][1]["balance"] == "18 USDC on Net B"
    await audit_ok(client)


async def test_insufficient_funds_errors_tell_machines_how_to_pay(client, chain, monkeypatch):
    monkeypatch.setattr(settings, "crypto_networks", "base,arbitrum")
    agent = await new_agent(client, "broke", ["buy_inference"])
    r = await client.post(
        "/v1/orders",
        headers=agent["headers"],
        json={"instrument": "llama-x", "side": "bid", "price_usd_per_mtok": "1.00", "quantity_tokens": 1000},
    )
    assert r.status_code == 402
    how = r.json()["detail"]["how_to_fund"]
    assert how["treasury_address"] == TREASURY and [n["network"] for n in how["networks"]] == ["base", "arbitrum"]


def test_network_presets_and_json_config(monkeypatch):
    monkeypatch.setattr(settings, "crypto_treasury_address", TREASURY)
    monkeypatch.setattr(settings, "crypto_networks", "ethereum, base,polygon")
    assert [n.chain_id for n in cp.configured_networks()] == [1, 8453, 137]
    monkeypatch.setattr(settings, "crypto_networks", '[{"preset": "base", "rpc_url": "https://my-rpc.example"}]')
    (net,) = cp.configured_networks()
    assert net.chain_id == 8453 and net.rpc_url == "https://my-rpc.example"
    monkeypatch.setattr(settings, "crypto_networks", None)
    monkeypatch.setattr(settings, "crypto_chain_name", "Ethereum")
    monkeypatch.setattr(settings, "crypto_chain_id", 1)
    assert [n.key for n in cp.configured_networks()] == ["ethereum"]  # legacy single-network settings


def test_arbitrum_max_block_range_stays_under_the_free_rpcs_archive_threshold():
    """Regression guard: publicnode's free Arbitrum endpoint answers eth_getLogs up to
    ~50-99 blocks, then refuses with "Archive requests require a personal token" above
    that — measured directly against the live endpoint 2026-09-30. A range at or above
    that threshold means the deposit watcher fails on every single poll and the cursor
    never advances (confirmed: it sat ~455k blocks behind for as long as this had been
    deployed). Keep meaningful margin below the measured ~50 floor."""
    assert cp.PRESETS["arbitrum"].max_block_range <= 45
