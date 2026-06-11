# apps/api/routes/analytics.py

from fastapi import APIRouter, Query
from datetime import datetime, timedelta
from sqlalchemy import select, func, case, and_
from infrastructure.db.connection import AsyncSessionLocal
from infrastructure.db.models import (
    Conversation, QualityEvaluation, ToolExecution,
    LLMCall, StateTransition
)
from infrastructure.cache.redis_cache import redis_cache

router = APIRouter()


@router.get("/analytics/overview")
async def get_overview(
    hours: int = Query(default=24, description="Lookback window in hours")
):
    """
    High-level dashboard metrics.
    Resolution rate, escalation rate, avg quality, token usage.
    """
    since = datetime.utcnow() - timedelta(hours=hours)

    async with AsyncSessionLocal() as session:
        # Total conversations
        total_q = await session.execute(
            select(func.count(Conversation.id)).where(
                Conversation.started_at >= since
            )
        )
        total = total_q.scalar() or 0

        # Escalated conversations
        escalated_q = await session.execute(
            select(func.count(Conversation.id)).where(
                and_(
                    Conversation.started_at >= since,
                    Conversation.escalated == True
                )
            )
        )
        escalated = escalated_q.scalar() or 0

        # Average quality scores
        avg_q = await session.execute(
            select(
                func.avg(QualityEvaluation.factual_accuracy),
                func.avg(QualityEvaluation.helpfulness),
                func.avg(QualityEvaluation.policy_compliance),
                func.avg(QualityEvaluation.tool_correctness),
                func.avg(QualityEvaluation.conversation_flow),
            ).where(QualityEvaluation.evaluated_at >= since)
        )
        avgs = avg_q.one_or_none()

        # Total tokens
        tokens_q = await session.execute(
            select(
                func.sum(LLMCall.prompt_tokens),
                func.sum(LLMCall.completion_tokens),
                func.count(LLMCall.id)
            ).where(LLMCall.called_at >= since)
        )
        token_row = tokens_q.one_or_none()

    resolved = total - escalated
    resolution_rate = round(resolved / total, 3) if total > 0 else 0.0

    return {
        "period_hours": hours,
        "total_conversations": total,
        "resolved": resolved,
        "escalated": escalated,
        "resolution_rate": resolution_rate,
        "escalation_rate": round(escalated / total, 3) if total > 0 else 0.0,
        "avg_quality_scores": {
            "factual_accuracy": round(float(avgs[0] or 0), 2),
            "helpfulness": round(float(avgs[1] or 0), 2),
            "policy_compliance": round(float(avgs[2] or 0), 2),
            "tool_correctness": round(float(avgs[3] or 0), 2),
            "conversation_flow": round(float(avgs[4] or 0), 2),
        } if avgs and avgs[0] else {},
        "token_usage": {
            "prompt_tokens": int(token_row[0] or 0) if token_row else 0,
            "completion_tokens": int(token_row[1] or 0) if token_row else 0,
            "total_llm_calls": int(token_row[2] or 0) if token_row else 0,
        }
    }


@router.get("/analytics/intents")
async def get_intent_distribution(
    hours: int = Query(default=24, description="Lookback window in hours")
):
    """
    Breakdown of conversation volume by intent.
    """
    since = datetime.utcnow() - timedelta(hours=hours)

    async with AsyncSessionLocal() as session:
        # Get state transitions where agent responsible logged the intent
        results = await session.execute(
            select(
                StateTransition.metadata_["primary_intent"].label("intent"),
                func.count(StateTransition.id)
            ).where(
                and_(
                    StateTransition.triggered_at >= since,
                    StateTransition.from_state == "INTENT_CLASSIFICATION"
                )
            ).group_by("intent")
        )
        rows = results.all()

    return {
        "period_hours": hours,
        "intents": [
            {"intent": row[0], "count": row[1]}
            for row in rows
        ]
    }


@router.get("/analytics/tools")
async def get_tool_performance(
    hours: int = Query(default=24, description="Lookback window in hours")
):
    """
    Tool success rates and latency.
    """
    since = datetime.utcnow() - timedelta(hours=hours)

    async with AsyncSessionLocal() as session:
        results = await session.execute(
            select(
                ToolExecution.tool_name,
                func.count(ToolExecution.id),
                func.sum(case((ToolExecution.success == True, 1), else_=0)),
                func.avg(ToolExecution.duration_ms)
            ).where(ToolExecution.executed_at >= since)
            .group_by(ToolExecution.tool_name)
        )
        rows = results.all()

    tools = []
    for row in rows:
        total = row[1]
        successes = row[2] or 0
        tools.append({
            "tool_name": row[0],
            "total_calls": total,
            "successes": successes,
            "failures": total - successes,
            "success_rate": round(successes / total, 3) if total > 0 else 0,
            "avg_latency_ms": round(float(row[3] or 0), 1)
        })

    return {"period_hours": hours, "tools": tools}


@router.get("/analytics/cache")
async def get_cache_stats():
    """Redis cache hit/miss statistics."""
    stats = await redis_cache.get_stats()
    return stats
