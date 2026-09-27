"""Double-entry micro-transaction ledger.

Accounts:
  agent:<id>:available   spendable balance
  agent:<id>:escrow      funds locked behind open bids and active allocations
  house:fees             clearing fees earned by the platform
  house:sandbox_mint     source of faucet money (goes negative by design)

Every posting is a set of legs that must sum to zero. Callers must hold row
locks on every agent touched (see lock_agents), so cached balances on the
Agent row cannot race.
"""

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .models import Agent, TradeLedger, new_id

AVAILABLE, ESCROW = "available", "escrow"
HOUSE_FEES = "house:fees"
HOUSE_MINT = "house:sandbox_mint"

_BUCKET_ATTR = {AVAILABLE: "balance_available_nanos", ESCROW: "balance_escrow_nanos"}


class LedgerError(Exception):
    pass


class InsufficientFunds(LedgerError):
    pass


class LedgerImbalance(LedgerError):
    pass


@dataclass(frozen=True)
class Leg:
    account: str
    amount: int
    agent_id: str | None = None
    bucket: str | None = None


def agent_leg(agent_id: str, bucket: str, amount: int) -> Leg:
    return Leg(f"agent:{agent_id}:{bucket}", amount, agent_id, bucket)


def house_leg(account: str, amount: int) -> Leg:
    return Leg(account, amount)


async def lock_agents(session: AsyncSession, agent_ids: Iterable[str]) -> dict[str, Agent]:
    """SELECT ... FOR UPDATE in a deterministic order to avoid deadlocks."""
    ids = sorted(set(agent_ids))
    if not ids:
        return {}
    rows = await session.execute(select(Agent).where(Agent.id.in_(ids)).order_by(Agent.id).with_for_update())
    agents = {a.id: a for a in rows.scalars()}
    missing = set(ids) - agents.keys()
    if missing:
        raise LedgerError(f"unknown agents {sorted(missing)}")
    return agents


def post(
    session: AsyncSession,
    agents: dict[str, Agent],
    legs: Iterable[Leg],
    *,
    kind: str,
    ref_type: str,
    ref_id: str,
    memo: str | None = None,
) -> str | None:
    legs = [leg for leg in legs if leg.amount != 0]
    if not legs:
        return None
    if sum(leg.amount for leg in legs) != 0:
        raise LedgerImbalance(f"{kind} legs do not balance: {legs}")

    deltas: dict[tuple[str, str], int] = defaultdict(int)
    for leg in legs:
        if leg.agent_id is not None:
            if leg.agent_id not in agents:
                raise LedgerError(f"agent {leg.agent_id} not locked for {kind}")
            deltas[(leg.agent_id, _BUCKET_ATTR[leg.bucket])] += leg.amount
    for (agent_id, attr), delta in deltas.items():
        if getattr(agents[agent_id], attr) + delta < 0:
            raise InsufficientFunds(f"{kind}: {agent_id} {attr} would go negative")
    for (agent_id, attr), delta in deltas.items():
        setattr(agents[agent_id], attr, getattr(agents[agent_id], attr) + delta)

    tx_id = new_id("tx")
    session.add_all(
        TradeLedger(
            tx_id=tx_id,
            account=leg.account,
            agent_id=leg.agent_id,
            amount_nanos=leg.amount,
            kind=kind,
            ref_type=ref_type,
            ref_id=ref_id,
            memo=memo,
        )
        for leg in legs
    )
    return tx_id


def mint(session, agents, agent_id: str, amount: int, *, memo: str) -> str | None:
    return post(
        session,
        agents,
        [house_leg(HOUSE_MINT, -amount), agent_leg(agent_id, AVAILABLE, amount)],
        kind="faucet",
        ref_type="agent",
        ref_id=agent_id,
        memo=memo,
    )


def lock_escrow(session, agents, agent_id: str, amount: int, *, ref_type: str, ref_id: str) -> str | None:
    return post(
        session,
        agents,
        [agent_leg(agent_id, AVAILABLE, -amount), agent_leg(agent_id, ESCROW, amount)],
        kind="escrow_lock",
        ref_type=ref_type,
        ref_id=ref_id,
    )


def release_escrow(session, agents, agent_id: str, amount: int, *, ref_type: str, ref_id: str, memo: str) -> str | None:
    return post(
        session,
        agents,
        [agent_leg(agent_id, ESCROW, -amount), agent_leg(agent_id, AVAILABLE, amount)],
        kind="escrow_release",
        ref_type=ref_type,
        ref_id=ref_id,
        memo=memo,
    )


def settle_delivery(
    session, agents, *, buyer_id: str, seller_id: str, gross: int, fee: int, trade_id: str, memo: str
) -> str | None:
    """Pay the seller out of the buyer's escrow for delivered tokens, minus fee."""
    return post(
        session,
        agents,
        [
            agent_leg(buyer_id, ESCROW, -gross),
            agent_leg(seller_id, AVAILABLE, gross - fee),
            house_leg(HOUSE_FEES, fee),
        ],
        kind="settlement",
        ref_type="trade",
        ref_id=trade_id,
        memo=memo,
    )
