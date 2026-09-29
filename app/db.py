from collections.abc import AsyncIterator
from datetime import datetime, timezone

from sqlalchemy import event, inspect
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from .config import settings


class Base(DeclarativeBase):
    pass


def _engine_kwargs(url: str) -> dict:
    if url.startswith("sqlite"):
        return {"connect_args": {"timeout": 30}}
    return {
        "pool_size": settings.db_pool_size,
        "max_overflow": settings.db_max_overflow,
        "pool_pre_ping": True,
    }


engine = create_async_engine(settings.database_url, **_engine_kwargs(settings.database_url))

if engine.dialect.name == "sqlite":
    # SQLite ignores SELECT ... FOR UPDATE. Take the database write lock at
    # BEGIN so read-modify-write transactions serialise exactly like the row
    # locks do on PostgreSQL (dev/test only; production runs PostgreSQL).
    @event.listens_for(engine.sync_engine, "connect")
    def _sqlite_connect(dbapi_connection, _record):
        dbapi_connection.isolation_level = None

    @event.listens_for(engine.sync_engine, "begin")
    def _sqlite_begin(conn):
        conn.exec_driver_sql("BEGIN IMMEDIATE")


SessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def get_session() -> AsyncIterator[AsyncSession]:
    async with SessionLocal() as session:
        yield session


async def init_models() -> None:
    # Schema bootstrap for the MVP. A production deployment would run Alembic
    # migrations here instead of create_all.
    from . import models  # noqa: F401  (registers tables on Base.metadata)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.run_sync(_add_missing_columns)


# Forward-only column additions for tables that already exist in a deployed
# database (create_all never alters existing tables). Replace with Alembic
# migrations once the schema changes more often.
_COLUMN_ADDITIONS = [("crypto_payouts", "chain_id", "INTEGER")]


def _add_missing_columns(sync_conn) -> None:
    insp = inspect(sync_conn)
    for table, column, ddl in _COLUMN_ADDITIONS:
        if insp.has_table(table) and column not in {c["name"] for c in insp.get_columns(table)}:
            sync_conn.exec_driver_sql(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def aware(dt: datetime | None) -> datetime | None:
    """SQLite hands back naive datetimes; everything we store is UTC."""
    if dt is not None and dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt
