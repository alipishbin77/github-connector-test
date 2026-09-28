import json

import httpx
import pytest
from eth_account import Account
from eth_account.messages import encode_defunct

from app import crypto_payments as cp
from app.config import settings

from .helpers import audit_ok, me, new_agent

TREASURY = "0x" + "7e" * 20
TOKEN = settings.crypto_token_address.lower()
ADMIN = "test-admin-token-0123456789abcdef"


def topic(addr: str) -> str:
    return "0x" + "0" * 24 + addr[2:].lower()


class FakeChain:
    """In-memory EVM JSON-RPC node: blocks, Transfer logs, receipts, balanceOf."""

    def __init__(self):
        self.head = 1_000
        self.logs: list[dict] = []
        self.receipts: dict[str, dict] = {}
        self.balance_units = 0

    def transfer_in(self, sender: str, units: int, block: int, tx: str, index: int = 0) -> None:
        self.logs.append(
            {
                "address": TOKEN,
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
            "logs": [{"address": TOKEN, "topics": [cp.TRANSFER_TOPIC, topic(TREASURY), topic(to)], "data": hex(units)}],
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
        else:
            return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "error": {"message": "unsupported"}})
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": result})


@pytest.fixture
async def chain(app, monkeypatch):
    fake = FakeChain()
    monkeypatch.setattr(settings, "crypto_treasury_address", TREASURY)
    monkeypatch.setattr(settings, "crypto_confirmations", 5)
    monkeypatch.setattr(settings, "admin_token", ADMIN)
    monkeypatch.setattr(settings, "crypto_chain_id", 900_000 + len(fake.logs) + id(fake) % 99_999)  # own cursor per test
    client = cp.EvmClient(httpx.AsyncClient(transport=httpx.MockTransport(fake.handle)), "http://rpc.test")
    monkeypatch.setattr(app.state, "chain", client, raising=False)
    watcher = cp.DepositWatcher(client)
    monkeypatch.setattr(app.state, "deposit_watcher", watcher, raising=False)
    fake.watcher = watcher
    yield fake
    await client.aclose()


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
    assert [d["status"] for d in deposits] == ["credited"] and deposits[0]["amount"] == "25 USDC"
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
    monkeypatch.setattr(app.state, "chain", None, raising=False)
    agent = await new_agent(client, "x", ["buy_inference"])
    assert (await client.get("/v1/billing/crypto", headers=agent["headers"])).status_code == 503
