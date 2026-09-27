"""Redis-backed continuous double auction with price/time (FIFO) priority.

Book layout (per instrument, e.g. "llama-3.1-70b-instruct"):

  {p}:book:{instrument}:asks   ZSET score = price_npt          (lowest first)
  {p}:book:{instrument}:bids   ZSET score = -price_npt         (highest first)
  member = "<seq:020d>:<order_id>" — equal scores sort lexicographically by
  member, so the zero-padded sequence number gives FIFO within a price level.

  {p}:order:{order_id}         HASH agent, remaining, price, expires_at, book, member, side
  {p}:expiry                   ZSET score = expires_at_ms, member = order_id
  {p}:stream:matches           STREAM of match/expiry events (durable fill log)

Matching runs inside a single Lua script, so it is atomic and serialised by
Redis without any application-level locks: any number of API workers can
submit orders concurrently. The script appends one event per match to a Redis
Stream; app/settlement.py applies each event to PostgreSQL exactly once
(inline on the request path, with a consumer-group worker as the crash-safe
fallback). PostgreSQL remains the source of truth; reconcile() rebuilds the
book from it if Redis loses data.

Note: the scripts derive order-hash keys from arguments, which is fine for a
single Redis primary but would need hash tags to run on Redis Cluster.
"""

import json
import time
from dataclasses import dataclass, field

from redis.asyncio import Redis

from .models import ASK, BID, Order

MATCH_LUA = r"""
local opp, own, expiry, stream = KEYS[1], KEYS[2], KEYS[3], KEYS[4]
local taker_id    = ARGV[1]
local taker_agent = ARGV[2]
local limit       = ARGV[3]
local remaining   = tonumber(ARGV[4])
local now         = tonumber(ARGV[5])
local max_fills   = tonumber(ARGV[6])
local rest        = ARGV[7]
local own_score   = ARGV[8]
local own_member  = ARGV[9]
local expires_at  = ARGV[10]
local price       = ARGV[11]
local instrument  = ARGV[12]
local side        = ARGV[13]
local prefix      = ARGV[14]
local maxlen      = ARGV[15]

local fills, expired = {}, {}
local nfills, skip = 0, 0

while remaining > 0 and nfills < max_fills do
  local batch = redis.call('ZRANGEBYSCORE', opp, '-inf', limit, 'LIMIT', skip, 64)
  if #batch == 0 then break end
  for _, member in ipairs(batch) do
    if remaining <= 0 or nfills >= max_fills then break end
    local oid = string.match(member, '^%d+:(.+)$')
    local okey = prefix .. oid
    local o = redis.call('HMGET', okey, 'agent', 'remaining', 'price', 'expires_at')
    if not o[1] then
      -- stale index entry
      redis.call('ZREM', opp, member)
      redis.call('ZREM', expiry, oid)
    elseif tonumber(o[4]) > 0 and tonumber(o[4]) <= now then
      redis.call('ZREM', opp, member)
      redis.call('ZREM', expiry, oid)
      redis.call('DEL', okey)
      table.insert(expired, {oid, o[2]})
    elseif o[1] == taker_agent then
      -- self-trade prevention: leave the maker resting, look past it
      skip = skip + 1
    else
      local avail = tonumber(o[2])
      local q = math.min(avail, remaining)
      remaining = remaining - q
      avail = avail - q
      if avail == 0 then
        redis.call('ZREM', opp, member)
        redis.call('ZREM', expiry, oid)
        redis.call('DEL', okey)
      else
        redis.call('HSET', okey, 'remaining', string.format('%d', avail))
      end
      nfills = nfills + 1
      fills[nfills] = {oid, o[1], o[3], string.format('%d', q)}
    end
  end
end

local rested = 0
if remaining > 0 and rest == '1' then
  local okey = prefix .. taker_id
  redis.call('HSET', okey, 'agent', taker_agent, 'remaining', string.format('%d', remaining),
             'price', price, 'expires_at', expires_at, 'book', own, 'member', own_member, 'side', side)
  redis.call('ZADD', own, own_score, own_member)
  if tonumber(expires_at) > 0 then
    redis.call('ZADD', expiry, expires_at, taker_id)
  end
  rested = 1
end

local payload = cjson.encode({
  taker = taker_id, side = side, instrument = instrument,
  remaining = string.format('%d', remaining), rested = rested,
  fills = fills, expired = expired, ts = now
})
local id = redis.call('XADD', stream, 'MAXLEN', '~', maxlen, '*', 'type', 'match', 'payload', payload)
return {id, payload}
"""

CANCEL_LUA = r"""
local o = redis.call('HMGET', KEYS[2], 'remaining', 'book', 'member')
if not o[1] then return -1 end
redis.call('ZREM', o[2], o[3])
redis.call('ZREM', KEYS[1], ARGV[1])
redis.call('DEL', KEYS[2])
return tonumber(o[1])
"""

SWEEP_LUA = r"""
local expiry, stream = KEYS[1], KEYS[2]
local now, prefix, limit, maxlen = ARGV[1], ARGV[2], ARGV[3], ARGV[4]
local ids = redis.call('ZRANGEBYSCORE', expiry, '-inf', now, 'LIMIT', 0, limit)
local expired = {}
for _, oid in ipairs(ids) do
  local okey = prefix .. oid
  local o = redis.call('HMGET', okey, 'remaining', 'book', 'member')
  redis.call('ZREM', expiry, oid)
  if o[1] then
    redis.call('ZREM', o[2], o[3])
    redis.call('DEL', okey)
    table.insert(expired, {oid, o[1]})
  end
end
if #expired == 0 then return false end
local payload = cjson.encode({taker = '', expired = expired, fills = {}, ts = tonumber(now)})
local id = redis.call('XADD', stream, 'MAXLEN', '~', maxlen, '*', 'type', 'expiry', 'payload', payload)
return {id, payload}
"""


@dataclass(frozen=True)
class Fill:
    maker_order_id: str
    maker_agent_id: str
    price_npt: int
    quantity: int


@dataclass
class MatchEvent:
    event_id: str
    type: str
    taker_order_id: str
    remaining: int
    rested: bool
    fills: list[Fill] = field(default_factory=list)
    expired: list[tuple[str, int]] = field(default_factory=list)

    @classmethod
    def from_payload(cls, event_id: str, type_: str, payload: str | dict) -> "MatchEvent":
        data = json.loads(payload) if isinstance(payload, str) else payload

        def as_list(value):  # cjson encodes an empty Lua table as {}
            return value if isinstance(value, list) else []

        return cls(
            event_id=event_id,
            type=type_,
            taker_order_id=data.get("taker") or "",
            remaining=int(data.get("remaining", 0)),
            rested=bool(data.get("rested", 0)),
            fills=[Fill(f[0], f[1], int(f[2]), int(f[3])) for f in as_list(data.get("fills"))],
            expired=[(e[0], int(e[1])) for e in as_list(data.get("expired"))],
        )


def _now_ms() -> int:
    return int(time.time() * 1000)


class MatchingEngine:
    def __init__(self, redis: Redis, prefix: str = "aether", *, max_fills: int = 256, stream_maxlen: int = 1_000_000):
        self.redis = redis
        self.prefix = prefix
        self.max_fills = max_fills
        self.stream_maxlen = stream_maxlen
        self.order_prefix = f"{prefix}:order:"
        self.expiry_key = f"{prefix}:expiry"
        self.stream_key = f"{prefix}:stream:matches"
        self.seq_key = f"{prefix}:seq"
        self._match = redis.register_script(MATCH_LUA)
        self._cancel = redis.register_script(CANCEL_LUA)
        self._sweep = redis.register_script(SWEEP_LUA)

    # ------------------------------------------------------------------ keys
    def book_key(self, instrument: str, side: str) -> str:
        return f"{self.prefix}:book:{instrument}:{'bids' if side == BID else 'asks'}"

    @staticmethod
    def member(order: Order) -> str:
        return f"{order.seq:020d}:{order.id}"

    @staticmethod
    def score(side: str, price_npt: int) -> int:
        return -price_npt if side == BID else price_npt

    async def next_seq(self) -> int:
        return int(await self.redis.incr(self.seq_key))

    # ------------------------------------------------------------- commands
    async def submit(self, order: Order, *, rest: bool) -> MatchEvent:
        """Match `order` as taker against the opposite book, then (if `rest`)
        leave any remainder resting in its own book. Atomic."""
        opposite = ASK if order.side == BID else BID
        # A bid crosses asks priced <= its limit; an ask crosses bids priced >= its
        # limit, i.e. bid scores (-price) <= -limit.
        limit_score = order.price_npt if order.side == BID else -order.price_npt
        expires_ms = int(order.expires_at.timestamp() * 1000) if order.expires_at else 0
        event_id, payload = await self._match(
            keys=[
                self.book_key(order.instrument, opposite),
                self.book_key(order.instrument, order.side),
                self.expiry_key,
                self.stream_key,
            ],
            args=[
                order.id,
                order.agent_id,
                limit_score,
                order.open_tokens,
                _now_ms(),
                self.max_fills,
                "1" if rest else "0",
                self.score(order.side, order.price_npt),
                self.member(order),
                expires_ms,
                order.price_npt,
                order.instrument,
                order.side,
                self.order_prefix,
                self.stream_maxlen,
            ],
        )
        return MatchEvent.from_payload(event_id, "match", payload)

    async def cancel(self, order_id: str) -> int | None:
        """Pull an order off the book. Returns the tokens that were still
        resting, or None if it was not on the book (already filled/expired)."""
        remaining = await self._cancel(keys=[self.expiry_key, self.order_prefix + order_id], args=[order_id])
        return None if int(remaining) < 0 else int(remaining)

    async def sweep_expired(self, limit: int = 500) -> MatchEvent | None:
        """Remove expired resting orders from every book in one atomic step."""
        result = await self._sweep(
            keys=[self.expiry_key, self.stream_key],
            args=[_now_ms(), self.order_prefix, limit, self.stream_maxlen],
        )
        if not result:
            return None
        event_id, payload = result
        return MatchEvent.from_payload(event_id, "expiry", payload)

    async def is_resting(self, order_id: str) -> bool:
        return bool(await self.redis.exists(self.order_prefix + order_id))

    async def restore(self, order: Order) -> None:
        """Re-insert an order from PostgreSQL (used by reconcile after Redis data loss)."""
        key = self.order_prefix + order.id
        book = self.book_key(order.instrument, order.side)
        expires_ms = int(order.expires_at.timestamp() * 1000) if order.expires_at else 0
        pipe = self.redis.pipeline(transaction=True)
        pipe.hset(
            key,
            mapping={
                "agent": order.agent_id,
                "remaining": order.open_tokens,
                "price": order.price_npt,
                "expires_at": expires_ms,
                "book": book,
                "member": self.member(order),
                "side": order.side,
            },
        )
        pipe.zadd(book, {self.member(order): self.score(order.side, order.price_npt)})
        if expires_ms:
            pipe.zadd(self.expiry_key, {order.id: expires_ms})
        await pipe.execute()

    async def depth(self, instrument: str, levels: int = 10, scan: int = 500) -> dict:
        """Aggregated price levels: {"bids": [(price_npt, tokens, orders)], "asks": [...]}."""
        out = {}
        now = _now_ms()
        for side in (BID, ASK):
            members = await self.redis.zrange(self.book_key(instrument, side), 0, scan - 1)
            pipe = self.redis.pipeline(transaction=False)
            for m in members:
                pipe.hmget(self.order_prefix + m.split(":", 1)[1], "remaining", "price", "expires_at")
            rows = await pipe.execute() if members else []
            agg: dict[int, list[int]] = {}
            for remaining, price, expires_at in rows:
                if remaining is None or (int(expires_at) and int(expires_at) <= now):
                    continue
                level = agg.setdefault(int(price), [0, 0])
                level[0] += int(remaining)
                level[1] += 1
            ordered = sorted(agg.items(), reverse=(side == BID))[:levels]
            out["bids" if side == BID else "asks"] = [(p, q, n) for p, (q, n) in ordered]
        return out
