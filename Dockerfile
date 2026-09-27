# syntax=docker/dockerfile:1
# One image serves the clearinghouse and both simulator agents.

FROM python:3.12-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
COPY requirements.txt /tmp/requirements.txt
RUN python -m venv /opt/venv && /opt/venv/bin/pip install -r /tmp/requirements.txt

FROM python:3.12-slim
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
RUN useradd --create-home --uid 10001 aether && mkdir -p /data && chown aether:aether /data
COPY --from=builder /opt/venv /opt/venv
WORKDIR /srv
COPY app ./app
USER aether
EXPOSE 8000
# uvloop + httptools; one worker per container — scale with replicas, not forks.
# PORT is honoured for PaaS hosts (Render, Railway, Fly, Cloud Run).
CMD exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}" --loop uvloop --http httptools \
    --no-access-log --timeout-graceful-shutdown 15 --proxy-headers --forwarded-allow-ips '*'
