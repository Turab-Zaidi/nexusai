# apps/api/routes/admin.py

from fastapi import APIRouter, Query, HTTPException
from datetime import datetime, timedelta
from sqlalchemy import select, func, desc, and_
from infrastructure.db.connection import AsyncSessionLocal
from infrastructure.db.models import (
    Conversation, StateTransition, QualityEvaluation,
    ToolExecution, LLMCall
)

router = APIRouter()


@router.get("/admin/conversations")
async def list_conversations(
    hours: int = Query(default=24, description="Lookback window"),
    escalated_only: bool = Query(default=False),
    limit: int = Query(default=50, le=200),
    offset: int = Query(default=0)
):
    """
    List conversations with filtering.
    """
    since = datetime.utcnow() - timedelta(hours=hours)

    async with AsyncSessionLocal() as session:
        query = select(Conversation).where(
            Conversation.started_at >= since
        )

        if escalated_only:
            query = query.where(Conversation.escalated == True)

        query = query.order_by(desc(Conversation.started_at)).limit(limit).offset(offset)

        result = await session.execute(query)
        conversations = result.scalars().all()

        # Count total for pagination
        count_q = select(func.count(Conversation.id)).where(
            Conversation.started_at >= since
        )
        if escalated_only:
            count_q = count_q.where(Conversation.escalated == True)
        total = (await session.execute(count_q)).scalar() or 0

    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "conversations": [
            {
                "id": str(conv.id),
                "user_id": str(conv.user_id) if conv.user_id else None,
                "channel": conv.channel,
                "current_state": conv.current_state,
                "resolution_status": conv.resolution_status,
                "escalated": conv.escalated,
                "escalation_reason": conv.escalation_reason,
                "total_turns": conv.total_turns,
                "started_at": str(conv.started_at) if conv.started_at else None,
                "ended_at": str(conv.ended_at) if conv.ended_at else None,
            }
            for conv in conversations
        ]
    }


@router.get("/admin/conversations/{conversation_id}")
async def get_conversation_detail(conversation_id: str):
    """
    Full conversation replay: transcript, state transitions,
    quality scores, tool calls, and LLM call details.
    """
    import uuid as uuid_mod
    try:
        conv_uuid = uuid_mod.UUID(conversation_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid conversation ID format")

    async with AsyncSessionLocal() as session:
        # Conversation
        conv_result = await session.execute(
            select(Conversation).where(Conversation.id == conv_uuid)
        )
        conv = conv_result.scalar_one_or_none()
        if not conv:
            raise HTTPException(status_code=404, detail="Conversation not found")

        # State transitions
        transitions_result = await session.execute(
            select(StateTransition)
            .where(StateTransition.conversation_id == conv_uuid)
            .order_by(StateTransition.triggered_at)
        )
        transitions = transitions_result.scalars().all()

        # Quality evaluations
        quality_result = await session.execute(
            select(QualityEvaluation)
            .where(QualityEvaluation.conversation_id == conv_uuid)
            .order_by(QualityEvaluation.evaluated_at)
        )
        quality_evals = quality_result.scalars().all()

        # Tool executions
        tools_result = await session.execute(
            select(ToolExecution)
            .where(ToolExecution.conversation_id == conv_uuid)
            .order_by(ToolExecution.executed_at)
        )
        tool_execs = tools_result.scalars().all()

        # LLM calls
        llm_result = await session.execute(
            select(LLMCall)
            .where(LLMCall.conversation_id == conv_uuid)
            .order_by(LLMCall.called_at)
        )
        llm_calls = llm_result.scalars().all()

    return {
        "conversation": {
            "id": str(conv.id),
            "user_id": str(conv.user_id) if conv.user_id else None,
            "channel": conv.channel,
            "current_state": conv.current_state,
            "resolution_status": conv.resolution_status,
            "escalated": conv.escalated,
            "escalation_reason": conv.escalation_reason,
            "total_turns": conv.total_turns,
            "total_cost_credits": conv.total_cost_credits,
            "started_at": str(conv.started_at) if conv.started_at else None,
            "ended_at": str(conv.ended_at) if conv.ended_at else None,
        },
        "state_transitions": [
            {
                "from_state": t.from_state,
                "to_state": t.to_state,
                "agent": t.agent_responsible,
                "triggered_at": str(t.triggered_at),
                "metadata": t.metadata_
            }
            for t in transitions
        ],
        "quality_evaluations": [
            {
                "factual_accuracy": q.factual_accuracy,
                "helpfulness": q.helpfulness,
                "policy_compliance": q.policy_compliance,
                "tool_correctness": q.tool_correctness,
                "conversation_flow": q.conversation_flow,
                "overall_pass": q.overall_pass,
                "revision_triggered": q.revision_triggered,
                "reasoning": q.judge_reasoning,
                "evaluated_at": str(q.evaluated_at)
            }
            for q in quality_evals
        ],
        "tool_executions": [
            {
                "tool_name": te.tool_name,
                "input": te.input_data,
                "output": te.output_data,
                "success": te.success,
                "error": te.error_message,
                "duration_ms": te.duration_ms,
                "executed_at": str(te.executed_at)
            }
            for te in tool_execs
        ],
        "llm_calls": [
            {
                "agent_name": lc.agent_name,
                "model": lc.model,
                "prompt_tokens": lc.prompt_tokens,
                "completion_tokens": lc.completion_tokens,
                "latency_ms": lc.latency_ms,
                "called_at": str(lc.called_at)
            }
            for lc in llm_calls
        ]
    }


@router.get("/admin/agents/performance")
async def get_agent_performance(
    hours: int = Query(default=24, description="Lookback window")
):
    """
    Per-agent quality score averages and call counts.
    """
    since = datetime.utcnow() - timedelta(hours=hours)

    async with AsyncSessionLocal() as session:
        results = await session.execute(
            select(
                LLMCall.agent_name,
                func.count(LLMCall.id),
                func.avg(LLMCall.latency_ms),
                func.sum(LLMCall.prompt_tokens + LLMCall.completion_tokens)
            ).where(LLMCall.called_at >= since)
            .group_by(LLMCall.agent_name)
        )
        rows = results.all()

    return {
        "period_hours": hours,
        "agents": [
            {
                "agent_name": row[0],
                "total_calls": row[1],
                "avg_latency_ms": round(float(row[2] or 0), 1),
                "total_tokens": int(row[3] or 0)
            }
            for row in rows
        ]
    }
