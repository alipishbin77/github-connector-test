"""Fixed-point money and price units.

All money is an integer number of nano-USD (1e-9 USD), so sub-cent
micro-transactions never touch floating point.

Prices are quoted to agents in USD per 1M tokens but stored as integer
nano-USD per token ("npt"): $0.35 / 1M tokens == 350 npt. The tick size is
therefore $0.001 per 1M tokens, and the cost of any fill is an exact integer
product price_npt * tokens — no rounding anywhere in the settlement path.
"""

from decimal import Decimal, InvalidOperation

NANOS_PER_USD = 1_000_000_000
TOKENS_PER_MTOK = 1_000_000
NPT_PER_USD_PER_MTOK = NANOS_PER_USD // TOKENS_PER_MTOK  # 1000


def usd_per_mtok_to_npt(price: Decimal | str | float) -> int:
    try:
        value = Decimal(str(price)) * NPT_PER_USD_PER_MTOK
    except InvalidOperation as exc:
        raise ValueError(f"invalid price {price!r}") from exc
    if value <= 0:
        raise ValueError("price must be positive")
    if value != value.to_integral_value():
        raise ValueError("price must be a multiple of the $0.001 / 1M-token tick")
    return int(value)


def npt_to_usd_per_mtok(npt: int) -> str:
    return str((Decimal(npt) / NPT_PER_USD_PER_MTOK).normalize())


def usd_to_nanos(usd: Decimal | str | float) -> int:
    value = Decimal(str(usd)) * NANOS_PER_USD
    if value != value.to_integral_value():
        raise ValueError("amount has more precision than 1 nano-USD")
    return int(value)


def fmt_usd(nanos: int) -> str:
    return f"${Decimal(nanos) / NANOS_PER_USD:.9f}"
