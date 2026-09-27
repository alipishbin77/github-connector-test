"""Test harness: real Redis (a throwaway redis-server) + SQLite, app lifespan
running in-process, sellers replaced by an httpx MockTransport."""

import os
import shutil
import socket
import subprocess
import tempfile
import time
import uuid

import pytest

_tmp = tempfile.mkdtemp(prefix="aether-test-")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


_redis_proc = None
if os.environ.get("AETHER_TEST_REDIS_URL"):
    redis_url = os.environ["AETHER_TEST_REDIS_URL"]
elif shutil.which("redis-server"):
    port = _free_port()
    _redis_proc = subprocess.Popen(
        ["redis-server", "--port", str(port), "--save", "", "--appendonly", "no"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(0.3)
    redis_url = f"redis://127.0.0.1:{port}/0"
else:  # pragma: no cover
    raise RuntimeError("tests need redis-server on PATH or AETHER_TEST_REDIS_URL")

# Must be set before any app module is imported.
os.environ["AETHER_DATABASE_URL"] = f"sqlite+aiosqlite:///{_tmp}/test.db"
os.environ["AETHER_REDIS_URL"] = redis_url
os.environ["AETHER_REDIS_PREFIX"] = f"aethertest{uuid.uuid4().hex[:8]}"
os.environ["AETHER_RETRY_BACKOFF_BASE_S"] = "0.01"
os.environ["AETHER_SWEEP_INTERVAL_S"] = "0.2"
os.environ["AETHER_SANDBOX_MODE"] = "true"
os.environ["AETHER_ALLOW_PRIVATE_SELLER_URLS"] = "true"
os.environ["AETHER_REGISTRATIONS_PER_IP_PER_HOUR"] = "0"


def pytest_sessionfinish(session, exitstatus):
    if _redis_proc is not None:
        _redis_proc.terminate()
    shutil.rmtree(_tmp, ignore_errors=True)


@pytest.fixture(scope="session")
async def app():
    from app.main import app as fastapi_app

    async with fastapi_app.router.lifespan_context(fastapi_app):
        yield fastapi_app


@pytest.fixture(scope="session")
async def client(app):
    import httpx

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://aether.test") as c:
        yield c


@pytest.fixture
def instrument():
    return f"model-{uuid.uuid4().hex[:8]}"
