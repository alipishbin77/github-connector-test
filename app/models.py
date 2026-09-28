"""Relational source of truth: agent registry, orders, trades (allocations) and
the double-entry ledger. Redis only holds the live order-book index, which can
be rebuilt from these tables at any time.

Money columns are integer nano-USD; prices are integer nano-USD per token
(see app/units.py).
"""

import uuid
from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base, utcnow

BigIntPK = BigInteger().with_variant(Integer, "sqlite")  # SQLite only autoincrements INTEGER


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


# Order sides / types / states
BID, ASK = "bid", "ask"
LIMIT, MARKET = "limit", "market"
GTC, IOC = "gtc", "ioc"
ORDER_OPEN, ORDER_FILLED, ORDER_CANCELLED, ORDER_EXPIRED = "open", "filled", "cancelled", "expired"

# Trade (inference-unit allocation) states
TRADE_ACTIVE, TRADE_EXHAUSTED, TRADE_RELEASED, TRADE_EXPIRED = "active", "exhausted", "released", "expired"

# Inference job states
JOB_STREAMING, JOB_COMPLETED, JOB_FAILED = "streaming", "completed", "failed"
JOB_REJECTED = "rejected"  # never routed: escrow gate refused it


class Agent(Base):
    __tablename__ = "agents"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("agt"))
    name: Mapped[str] = mapped_column(String(128))
    client_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    client_secret_hash: Mapped[str] = mapped_column(String(256))
    allowed_scopes: Mapped[str] = mapped_column(String(256))  # space separated
    endpoint_url: Mapped[str | None] = mapped_column(String(512))  # seller inference webhook
    balance_available_nanos: Mapped[int] = mapped_column(BigInteger, default=0)
    balance_escrow_nanos: Mapped[int] = mapped_column(BigInteger, default=0)
    token_version: Mapped[int] = mapped_column(Integer, default=0)  # bump to revoke all JWTs
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    __table_args__ = (
        CheckConstraint("balance_available_nanos >= 0", name="ck_agent_available_nonneg"),
        CheckConstraint("balance_escrow_nanos >= 0", name="ck_agent_escrow_nonneg"),
    )

    @property
    def scopes(self) -> set[str]:
        return set(self.allowed_scopes.split())


class Order(Base):
    __tablename__ = "orders"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("ord"))
    agent_id: Mapped[str] = mapped_column(ForeignKey("agents.id"), index=True)
    instrument: Mapped[str] = mapped_column(String(64))
    side: Mapped[str] = mapped_column(String(4))
    order_type: Mapped[str] = mapped_column(String(8))
    time_in_force: Mapped[str] = mapped_column(String(3))
    # Limit price, or the protection cap/floor of a market order.
    price_npt: Mapped[int] = mapped_column(BigInteger)
    quantity_tokens: Mapped[int] = mapped_column(BigInteger)
    filled_tokens: Mapped[int] = mapped_column(BigInteger, default=0)
    cancelled_tokens: Mapped[int] = mapped_column(BigInteger, default=0)
    # Bids only: invariant escrow_nanos == price_npt * open_tokens.
    escrow_nanos: Mapped[int] = mapped_column(BigInteger, default=0)
    status: Mapped[str] = mapped_column(String(16), default=ORDER_OPEN)
    seq: Mapped[int] = mapped_column(BigInteger)  # time priority, from Redis INCR
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    __table_args__ = (
        CheckConstraint("quantity_tokens > 0", name="ck_order_qty_pos"),
        CheckConstraint("price_npt > 0", name="ck_order_price_pos"),
        CheckConstraint("filled_tokens + cancelled_tokens <= quantity_tokens", name="ck_order_fill_bounds"),
        CheckConstraint("escrow_nanos >= 0", name="ck_order_escrow_nonneg"),
        Index("ix_orders_status_instrument", "status", "instrument"),
    )

    @property
    def open_tokens(self) -> int:
        return self.quantity_tokens - self.filled_tokens - self.cancelled_tokens


class Trade(Base):
    """A fill. For the buyer it is also an *allocation*: the right to consume
    `tokens_total` inference units from `seller_id` at `price_npt`, fully
    pre-funded by `escrow_nanos`. The seller is paid only for tokens actually
    delivered through the proxy router."""

    __tablename__ = "trades"

    id: Mapped[str] = mapped_column(String(96), primary_key=True)
    event_id: Mapped[str] = mapped_column(String(64), index=True)
    instrument: Mapped[str] = mapped_column(String(64))
    bid_order_id: Mapped[str] = mapped_column(ForeignKey("orders.id"))
    ask_order_id: Mapped[str] = mapped_column(ForeignKey("orders.id"))
    buyer_id: Mapped[str] = mapped_column(ForeignKey("agents.id"), index=True)
    seller_id: Mapped[str] = mapped_column(ForeignKey("agents.id"), index=True)
    taker_side: Mapped[str] = mapped_column(String(4))
    price_npt: Mapped[int] = mapped_column(BigInteger)
    tokens_total: Mapped[int] = mapped_column(BigInteger)
    tokens_used: Mapped[int] = mapped_column(BigInteger, default=0)
    tokens_reserved: Mapped[int] = mapped_column(BigInteger, default=0)  # in-flight requests
    # Invariant while active: escrow_nanos == price_npt * (tokens_total - tokens_used).
    escrow_nanos: Mapped[int] = mapped_column(BigInteger)
    gross_settled_nanos: Mapped[int] = mapped_column(BigInteger, default=0)
    fee_nanos: Mapped[int] = mapped_column(BigInteger, default=0)
    status: Mapped[str] = mapped_column(String(16), default=TRADE_ACTIVE)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    __table_args__ = (
        CheckConstraint("tokens_used + tokens_reserved <= tokens_total", name="ck_trade_capacity"),
        CheckConstraint("tokens_reserved >= 0", name="ck_trade_reserved_nonneg"),
        CheckConstraint("escrow_nanos >= 0", name="ck_trade_escrow_nonneg"),
        Index("ix_trades_buyer_active", "buyer_id", "instrument", "status"),
    )

    @property
    def tokens_available(self) -> int:
        return self.tokens_total - self.tokens_used - self.tokens_reserved


class TradeLedger(Base):
    """Append-only double-entry journal. Every tx_id's legs sum to zero, and the
    cached balances on Agent are exactly the sum of that agent's legs."""

    __tablename__ = "trade_ledger"

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    tx_id: Mapped[str] = mapped_column(String(64), index=True)
    account: Mapped[str] = mapped_column(String(128), index=True)
    agent_id: Mapped[str | None] = mapped_column(String(64), index=True)
    amount_nanos: Mapped[int] = mapped_column(BigInteger)  # signed
    kind: Mapped[str] = mapped_column(String(32))
    ref_type: Mapped[str] = mapped_column(String(16))
    ref_id: Mapped[str] = mapped_column(String(96))
    memo: Mapped[str | None] = mapped_column(String(256))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class AppliedMatchEvent(Base):
    """Idempotency guard: each Redis match-stream event is applied exactly once."""

    __tablename__ = "applied_match_events"

    event_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    applied_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class InferenceJob(Base):
    __tablename__ = "inference_jobs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("job"))
    buyer_id: Mapped[str] = mapped_column(ForeignKey("agents.id"), index=True)
    instrument: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(16), default=JOB_STREAMING)
    finish_reason: Mapped[str | None] = mapped_column(String(32))
    tokens_requested: Mapped[int] = mapped_column(Integer)
    tokens_delivered: Mapped[int] = mapped_column(Integer, default=0)
    cost_nanos: Mapped[int] = mapped_column(BigInteger, default=0)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    failovers: Mapped[int] = mapped_column(Integer, default=0)
    segments: Mapped[str | None] = mapped_column(Text)  # JSON: per-seller delivery breakdown
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class ExternalPayment(Base):
    """A settled deposit from an external rail (Stripe Checkout). The primary
    key is the provider's object id, which makes webhook crediting idempotent."""

    __tablename__ = "external_payments"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    agent_id: Mapped[str] = mapped_column(ForeignKey("agents.id"), index=True)
    provider: Mapped[str] = mapped_column(String(16))
    amount_nanos: Mapped[int] = mapped_column(BigInteger)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class PayoutAccount(Base):
    """Seller's connected payout account (Stripe Connect Express)."""

    __tablename__ = "payout_accounts"

    agent_id: Mapped[str] = mapped_column(ForeignKey("agents.id"), primary_key=True)
    stripe_account_id: Mapped[str] = mapped_column(String(64), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Withdrawal(Base):
    __tablename__ = "withdrawals"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("wd"))
    agent_id: Mapped[str] = mapped_column(ForeignKey("agents.id"), index=True)
    amount_nanos: Mapped[int] = mapped_column(BigInteger)
    status: Mapped[str] = mapped_column(String(16))  # pending|paid|failed
    stripe_transfer_id: Mapped[str | None] = mapped_column(String(64))
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class LinkedWallet(Base):
    """An on-chain address an agent proved it controls (signed challenge).
    Deposits from it are credited to the agent; payouts may only go to it."""

    __tablename__ = "linked_wallets"

    address: Mapped[str] = mapped_column(String(42), primary_key=True)  # lowercase 0x...
    agent_id: Mapped[str] = mapped_column(ForeignKey("agents.id"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class CryptoDeposit(Base):
    """One token Transfer into the treasury. Primary key '<chain>:<tx>:<log index>'
    makes crediting idempotent across re-scans and replicas."""

    __tablename__ = "crypto_deposits"

    id: Mapped[str] = mapped_column(String(100), primary_key=True)
    tx_hash: Mapped[str] = mapped_column(String(66), index=True)
    from_address: Mapped[str] = mapped_column(String(42), index=True)
    amount_units: Mapped[int] = mapped_column(BigInteger)
    block_number: Mapped[int] = mapped_column(BigInteger)
    agent_id: Mapped[str | None] = mapped_column(ForeignKey("agents.id"), index=True)
    status: Mapped[str] = mapped_column(String(16))  # credited | unattributed
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ChainCursor(Base):
    __tablename__ = "chain_cursors"

    chain_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    last_block: Mapped[int] = mapped_column(BigInteger)


class CryptoPayout(Base):
    """A withdrawal to an agent's linked wallet. Debited in the ledger at request
    time; sent by the operator from the treasury; marked paid only after the
    transfer is verified on-chain."""

    __tablename__ = "crypto_payouts"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=lambda: new_id("cpo"))
    agent_id: Mapped[str] = mapped_column(ForeignKey("agents.id"), index=True)
    to_address: Mapped[str] = mapped_column(String(42))
    amount_units: Mapped[int] = mapped_column(BigInteger)
    status: Mapped[str] = mapped_column(String(16), index=True)  # pending | paid | cancelled
    tx_hash: Mapped[str | None] = mapped_column(String(66), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    paid_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
