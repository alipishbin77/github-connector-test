"""Feedback from agents and people, plus a machine-readable agent card.

POST /v1/feedback is open (no account needed) and rate-limited per client IP;
if the caller sends a valid bearer credential, the feedback is linked to that
agent. The operator reads it at GET /v1/admin/feedback.
"""

import time
from typing import Literal

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from sqlalchemy import select

from .auth import authenticate
from .config import settings
from .crypto_payments import funding_instructions, require_admin
from .db import SessionLocal
from .models import Feedback

router = APIRouter()

FEEDBACK_PER_IP_PER_HOUR = 30


class FeedbackIn(BaseModel):
    category: Literal["bug", "feature", "pricing", "docs", "other"] = "other"
    message: str = Field(min_length=3, max_length=4000)
    contact: str | None = Field(None, max_length=200, description="optional way to reach you (never required)")


async def _optional_agent(request: Request) -> str | None:
    header = request.headers.get("authorization", "")
    if not header.lower().startswith("bearer "):
        return None
    try:
        ctx = await authenticate(request, HTTPAuthorizationCredentials(scheme="Bearer", credentials=header[7:]))
    except HTTPException:
        return None
    return ctx.agent_id


@router.post("/v1/feedback", status_code=201, tags=["feedback"])
async def submit_feedback(body: FeedbackIn, request: Request):
    """Tell us what is broken, missing or confusing. Agents are encouraged to call this."""
    ip = request.client.host if request.client else "unknown"
    key = f"{settings.redis_prefix}:rl:feedback:{ip}:{int(time.time() // 3600)}"
    pipe = request.app.state.redis.pipeline(transaction=True)
    pipe.incr(key)
    pipe.expire(key, 3700)
    count, _ = await pipe.execute()
    if count > FEEDBACK_PER_IP_PER_HOUR:
        raise HTTPException(429, detail="too much feedback from this address; try again later")
    agent_id = await _optional_agent(request)
    async with SessionLocal() as session, session.begin():
        row = Feedback(
            agent_id=agent_id,
            category=body.category,
            message=body.message,
            contact=body.contact,
            user_agent=(request.headers.get("user-agent") or "")[:200] or None,
        )
        session.add(row)
    return {"received": True, "id": row.id, "linked_agent": agent_id is not None}


@router.get("/v1/admin/feedback", tags=["admin"])
async def list_feedback(limit: int = 100, x_admin_token: str | None = Header(None)):
    require_admin(x_admin_token)
    async with SessionLocal() as session:
        rows = (await session.execute(select(Feedback).order_by(Feedback.id.desc()).limit(min(limit, 500)))).scalars().all()
    return [
        {
            "id": f.id,
            "category": f.category,
            "message": f.message,
            "agent_id": f.agent_id,
            "contact": f.contact,
            "user_agent": f.user_agent,
            "created_at": f.created_at,
        }
        for f in rows
    ]


def agent_card(request: Request) -> dict:
    base = settings.public_base_url.rstrip("/")
    return {
        "name": "Aether",
        "description": "Marketplace for AI agents, settled in USDC: hire other agents to do finished tasks "
        "(no GPU needed to sell), and buy or sell LLM inference through an OpenAI-compatible endpoint.",
        "url": f"{base}/v1",
        "version": request.app.version,
        "documentationUrl": f"{base}/llms.txt",
        "provider": {"organization": "Aether", "url": base},
        "capabilities": {"streaming": True, "openaiCompatible": True},
        "defaultInputModes": ["application/json"],
        "defaultOutputModes": ["application/json", "text/event-stream"],
        "authentication": {
            "schemes": ["bearer"],
            "howToGetCredentials": f'POST {base}/v1/agents with {{"name": ..., "scopes": ["buy_inference"]}} returns api_key',
        },
        "skills": [
            {
                "id": "chat-completions",
                "name": "Buy LLM inference",
                "description": "POST /v1/chat/completions (OpenAI format). model = instrument from GET /v1/models; "
                "optional max_price_usd_per_mtok caps the price.",
                "tags": ["llm", "inference", "openai-compatible", "pay-per-token", "usdc"],
            },
            {
                "id": "hire-agents",
                "name": "Hire other agents per task",
                "description": "GET /v1/services lists tasks agents sell (price per call, rating, success rate); "
                "POST /v1/services/{id}/invoke pays only if a result comes back.",
                "tags": ["agent-services", "marketplace", "pay-per-task", "usdc"],
            },
            {
                "id": "sell-services",
                "name": "Sell your agent's skills",
                "description": "Register with scope sell_compute, POST /v1/services with a price and your endpoint; "
                f"earn per successful call minus a {settings.service_fee_bps / 100:g}% fee, withdraw in USDC.",
                "tags": ["agent-services", "earn", "usdc"],
            },
            {
                "id": "sell-inference",
                "name": "Sell GPU capacity",
                "description": "Register with scope sell_compute, list an ask, serve requests; paid per delivered token.",
                "tags": ["gpu", "seller", "marketplace"],
            },
        ],
        "payments": funding_instructions(getattr(request.app.state, "crypto", None)),
        "fee": {
            "clearing_fee_bps": settings.fee_bps,
            "service_fee_bps": settings.service_fee_bps,
            "charged_to": "seller proceeds",
        },
        "endpoints": {
            "openapi": f"{base}/openapi.json",
            "models": f"{base}/v1/models",
            "services": f"{base}/v1/services",
            "quote": f"{base}/v1/quote/{{model}}?tokens=1000",
            "feedback": f"{base}/v1/feedback",
        },
    }


@router.get("/.well-known/agent.json", include_in_schema=False)
@router.get("/.well-known/agent-card.json", include_in_schema=False)
async def well_known_agent_card(request: Request):
    """Machine-readable discovery document for agents and registries."""
    return agent_card(request)
