"""Runtime configuration. Every field can be overridden with an AETHER_* env var."""

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AETHER_", extra="ignore")

    # Storage
    database_url: str = "postgresql+asyncpg://aether:aether@localhost:5432/aether"
    db_pool_size: int = 20
    db_max_overflow: int = 20
    redis_url: str = "redis://localhost:6379/0"
    redis_max_connections: int = 200
    redis_prefix: str = "aether"

    # OAuth2 / JWT (RS256). If neither a PEM nor a key file is given, an
    # ephemeral key is generated at boot (fine for a single-process sandbox).
    jwt_issuer: str = "https://clearinghouse.aether.local"
    jwt_private_key_pem: str | None = None
    jwt_key_file: str | None = None
    access_token_ttl_s: int = 900
    delivery_token_ttl_s: int = 60

    # Market
    fee_bps: int = 100  # clearing fee charged on seller proceeds (1.00%)
    allocation_ttl_s: int = 3600  # how long bought inference units stay usable
    default_ask_ttl_s: int = 3600
    max_order_tokens: int = 10**12
    max_fills_per_match: int = 256
    # OpenAI-compatible endpoint defaults
    default_completion_tokens: int = 512
    default_max_price_usd_per_mtok: str = "5.00"  # auto-buy price cap when the agent sends none
    match_stream_maxlen: int = 1_000_000

    # Agent services marketplace (agents sell finished tasks, priced per call)
    service_fee_bps: int = 1000  # 10% of each successful call
    service_max_timeout_s: int = 120
    service_max_output_bytes: int = 1_000_000

    # Proxy router
    seller_connect_timeout_s: float = 3.0
    seller_read_timeout_s: float = 10.0  # max silence between two streamed chunks
    max_attempts_per_allocation: int = 2  # resume on the same seller before failing over
    max_failovers: int = 2
    retry_backoff_base_s: float = 0.25
    checkpoint_every_tokens: int = 16
    checkpoint_ttl_s: int = 3600
    max_tokens_per_request: int = 32_768
    http_max_connections: int = 1000

    # Background workers
    sweep_interval_s: float = 2.0

    # Real-money rails (Stripe). Unset => billing endpoints return 503.
    public_base_url: str = "http://localhost:8000"
    stripe_secret_key: str | None = None
    stripe_webhook_secret: str | None = None
    stripe_api_base: str = "https://api.stripe.com"
    min_deposit_usd: int = 5
    max_deposit_usd: int = 10_000
    min_withdrawal_usd: int = 10

    # Crypto rails (stablecoin on an EVM chain). Unset treasury => crypto endpoints return 503.
    # Defaults: native USDC on Base mainnet.
    crypto_treasury_address: str | None = None  # the platform wallet agents pay into (public address only)
    # Networks to accept: comma list of presets (ethereum,base,arbitrum,optimism,polygon) or a JSON list.
    # Unset => the single network described by the crypto_* fields below (legacy).
    crypto_networks: str | None = None
    crypto_rpc_url: str = "https://mainnet.base.org"
    crypto_chain_id: int = 8453
    crypto_chain_name: str = "Base"
    crypto_token_address: str = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
    crypto_token_symbol: str = "USDC"
    crypto_token_decimals: int = 6
    crypto_confirmations: int = 10
    crypto_poll_interval_s: float = 15.0
    crypto_start_block: int | None = None  # first scan block; default: current safe head at first run
    crypto_max_block_range: int = 500
    crypto_min_withdrawal_usd: int = 10

    # Security / sandbox
    # Production-safe defaults; docker-compose and tests opt into the sandbox explicitly.
    sandbox_mode: bool = False  # open registration + faucet + audit endpoint
    admin_token: str | None = None  # gates /v1/admin/*, and registration when open_registration is off
    open_registration: bool = True  # agents self-register (zero balance until they deposit)
    registrations_per_ip_per_hour: int = 20
    max_faucet_usd: float = 100.0
    allow_private_seller_urls: bool = False  # SSRF guard: sellers must be public https endpoints
    cors_origins: list[str] = ["*"]
    rate_limit_per_minute: int = 1200  # per agent, all authenticated endpoints; 0 disables

    @field_validator("database_url")
    @classmethod
    def _async_driver(cls, v: str) -> str:
        # Managed Postgres (Render, Heroku, Fly, Neon...) hands out postgres:// URLs.
        for prefix in ("postgres://", "postgresql://"):
            if v.startswith(prefix):
                return "postgresql+asyncpg://" + v[len(prefix) :]
        return v


settings = Settings()
