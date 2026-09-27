"""Runtime configuration. Every field can be overridden with an AETHER_* env var."""

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
    match_stream_maxlen: int = 1_000_000

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

    # Security / sandbox
    sandbox_mode: bool = True  # open registration + faucet + audit endpoint
    admin_token: str | None = None  # required for registration when sandbox_mode is off
    max_faucet_usd: float = 100.0
    allow_private_seller_urls: bool = True  # SSRF guard; set False in production
    cors_origins: list[str] = ["*"]


settings = Settings()
