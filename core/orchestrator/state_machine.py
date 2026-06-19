# core/orchestrator/state_machine.py

from langgraph.graph import StateGraph, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
import os
import json
import dataclasses

from .state import NexusState, ConversationStateEnum
from core.agents.input_classifier import InputClassifier
from core.agents.knowledge_agent import KnowledgeAgent
from core.agents.action_agent import ActionAgent
from core.quality.judge import QualityJudge
from infrastructure.cache.redis_cache import redis_cache


def create_graph():
    """
    Create the LangGraph state machine.
    This defines every possible state and transition.
    """

    graph = StateGraph(NexusState)

    # ── Add all nodes (states) ──────────────────────────

    graph.add_node("context_loader", context_loader_node)
    graph.add_node("intent_classification", intent_classification_node)
    graph.add_node("collecting_info", collecting_info_node)
    graph.add_node("knowledge_retrieval", knowledge_retrieval_node)
    graph.add_node("action_execution", action_execution_node)
    graph.add_node("quality_check", quality_check_node)
    graph.add_node("revision", revision_node)
    graph.add_node("escalation", escalation_node)

    # ── Set entry point ─────────────────────────────────

    graph.set_entry_point("context_loader")

    # ── Add edges (transitions) ─────────────────────────

    graph.add_edge("context_loader", "intent_classification")

    # From intent_classification → route based on AI analysis
    graph.add_conditional_edges(
        "intent_classification",
        route_after_classification,
        {
            "collecting_info": "collecting_info",
            "knowledge_retrieval": "knowledge_retrieval",
            "action_execution": "action_execution",
            "escalation": "escalation"
        }
    )

    graph.add_edge("collecting_info", "intent_classification")
    graph.add_edge("knowledge_retrieval", "quality_check")
    graph.add_edge("action_execution", "quality_check")

    # From quality_check → pass, revise, or escalate
    graph.add_conditional_edges(
        "quality_check",
        route_after_quality_check,
        {
            "end": END,
            "revision": "revision",
            "escalation": "escalation"
        }
    )

    graph.add_conditional_edges(
        "revision",
        route_after_revision,
        {
            "quality_check": "quality_check",
            "escalation": "escalation"
        }
    )

    graph.add_edge("escalation", END)

    # Compile with a checkpointer so that interrupt() can pause and resume
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


# ── Routing functions ────────────────────────────────────

def route_after_classification(state: NexusState) -> str:
    """Decide where to go after analyzing intent."""

    analyzed = state.analyzed_input
    if not analyzed:
        return "knowledge_retrieval"
        
    # ── Escalation checks (highest priority) ─────────────
    risk_flags = analyzed.risk_flags or []
    escalation_triggers = ["lawsuit_threat", "explicit_human_request", "hardship", "suicide_risk"]
    if any(flag in risk_flags for flag in escalation_triggers):
        return "escalation"

    intent = analyzed.primary_intent or "unclear"

    # ── FinTech Action Intents (need DB tools) ───────────
    action_intents = [
        "fee_waiver", "freeze_card", "unfreeze_card",
        "submit_dispute", "check_transaction",
        "report_fraud", "request_virtual_card", "account_info",
        "financial_analysis"
    ]

    # ── FinTech Knowledge Intents (need RAG) ─────────────
    knowledge_intents = [
        "policy_question", "check_fees", "general_inquiry",
        "check_dispute_status", "regulation_question", "complaint"
    ]

    # ── Route action intents ─────────────────────────────
    if intent in action_intents:
        entities = analyzed.entities
        print(f"[ROUTE] Action Intent: {intent}, Entities: {entities.model_dump() if entities else None}")
        # If user says "freeze my card" but we don't know which card
        if not (entities and entities.card_id) and intent in ["freeze_card", "unfreeze_card", "report_fraud"]:
            print("[ROUTE] Missing card_id for action intent -> collecting_info")
            return "collecting_info"
        # If user says "dispute this charge" but no transaction specified
        if not (entities and entities.transaction_id) and intent in ["submit_dispute", "fee_waiver"]:
            print("[ROUTE] Missing transaction_id for action intent -> collecting_info")
            return "collecting_info"
        print("[ROUTE] All entities present -> action_execution")
        return "action_execution"

    if intent in knowledge_intents:
        return "knowledge_retrieval"

    # Default to knowledge retrieval for unclear inputs
    return "knowledge_retrieval"


def route_after_quality_check(state: NexusState) -> str:
    """Decide what to do based on quality scores."""

    if state.quality_passed:
        return "end"

    revision_count = state.revision_count or 0
    scores = state.quality_scores or {}

    # Policy violation → immediate escalation
    if scores.get("policy_compliance", 5) < 4:
        return "escalation"

    # Two failed revisions → escalate
    if revision_count >= 2:
        return "escalation"

    return "revision"


def route_after_revision(state: NexusState) -> str:
    """After revision, try quality check again"""

    revision_count = state.revision_count or 0
    if revision_count >= 2:
        return "escalation"
    return "quality_check"


# Instantiate agents
input_classifier = InputClassifier("input_classifier", "fast")
knowledge_agent = KnowledgeAgent()
action_agent = ActionAgent()
quality_judge = QualityJudge()


# ── Node Implementations ─────────────────────────────────

async def context_loader_node(state: NexusState) -> dict:
    """
    Entry point. Retrieves user support tickets from SQLite
    to provide personalized context to downstream agents.
    """
    from infrastructure.db.connection import AsyncSessionLocal
    from infrastructure.db.models import SupportTicket
    from sqlalchemy import select
    
    user_id = state.user_id or "anonymous"
    
    past_tickets = []
    ticket_ids = []
    
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(SupportTicket)
            .where(SupportTicket.user_id == user_id)
            .order_by(SupportTicket.created_at.desc())
            .limit(3)
        )
        tickets = result.scalars().all()
        for t in tickets:
            ticket_ids.append(t.id)
            date_str = t.created_at.strftime('%Y-%m-%d') if t.created_at else "Unknown"
            past_tickets.append(f"- [{date_str}] Intent: {t.intent} | Status: {t.status} | Summary: {t.summary}")
            
    user_context_str = "Past Interactions:\\n" + "\\n".join(past_tickets) if past_tickets else "No past interactions."

    return {
        "current_state": ConversationStateEnum.CONTEXT_LOADER.value,
        "turn_count": (state.turn_count or 0) + 1,
        "user_context": {"memories": ticket_ids, "context_string": user_context_str}
    }


async def intent_classification_node(state: NexusState) -> dict:
    message = state.current_message or ""
    conversation_history = state.messages[-6:] if state.messages else []
    result = await input_classifier.run(message, conversation_history=conversation_history)
    
    analyzed_obj = result.output
    
    # Preserve overrides sent from the UI buttons
    if state.analyzed_input:
        if state.analyzed_input.intent_override:
            analyzed_obj.primary_intent = state.analyzed_input.intent_override
        if state.analyzed_input.entities:
            from core.agents.input_classifier import EntityResult
            if not analyzed_obj.entities:
                analyzed_obj.entities = EntityResult()
            if state.analyzed_input.entities.transaction_id:
                analyzed_obj.entities.transaction_id = state.analyzed_input.entities.transaction_id
            if state.analyzed_input.entities.card_id:
                analyzed_obj.entities.card_id = state.analyzed_input.entities.card_id
    
    print(f"[CLASSIFIER] Message: '{message}' -> Intent: '{analyzed_obj.primary_intent}' (confidence: {analyzed_obj.confidence})")
    return {
        "current_state": ConversationStateEnum.INTENT_CLASSIFICATION.value,
        "analyzed_input": analyzed_obj
    }


async def collecting_info_node(state: NexusState) -> dict:
    from infrastructure.llm.nim_client import nim_client

    message = state.current_message or ""
    analyzed = state.analyzed_input
    intent = analyzed.primary_intent if analyzed else "unclear"
    entities = analyzed.entities if analyzed else None

    # Figure out what FinTech field is missing
    missing_fields = []
    if not (entities and entities.card_id) and intent in ["freeze_card", "unfreeze_card", "report_fraud"]:
        missing_fields.append("which card (e.g. physical or virtual, or last 4 digits)")
    if not (entities and entities.transaction_id) and intent in ["submit_dispute", "fee_waiver"]:
        missing_fields.append("which transaction (e.g. merchant name or transaction ID)")

    if not missing_fields:
        missing_fields = ["more details about your request"]

    # Use LLM to generate a natural follow-up question
    response = await nim_client.complete(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a concise, professional Nexus Bank support agent. "
                    "Ask the customer for the missing detail in ONE short sentence. "
                    "Do not offer help, just ask the specific question."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Customer said: '{message}'\n"
                    f"Their intent is: {intent}\n"
                    f"Missing information needed: {', '.join(missing_fields)}\n"
                    "Ask them for the missing info."
                )
            }
        ],
        tier="fast",
        max_tokens=80
    )

    follow_up = response.get("content") or f"Could you clarify {', '.join(missing_fields)}?"

    # PAUSE the graph — send the question back to the user via the UI
    user_reply = interrupt(follow_up)

    # Graph resumes here once the user responds
    current_messages = state.messages or []
    current_messages.append({"role": "assistant", "content": follow_up})
    current_messages.append({"role": "user", "content": user_reply})

    return {
        "current_state": ConversationStateEnum.COLLECTING_INFO.value,
        "current_message": user_reply,
        "messages": current_messages
    }


async def knowledge_retrieval_node(state: NexusState) -> dict:
    message = state.current_message or ""
    analyzed = state.analyzed_input
    intent = analyzed.primary_intent if analyzed else "unclear"
    user_context = state.user_context or {}

    print(f"[KNOWLEDGE] Received intent: '{intent}' for message: '{message}'")

    cached_response = await redis_cache.get(intent=intent, query=message)
    if cached_response:
        return {
            "current_state": ConversationStateEnum.KNOWLEDGE_RETRIEVAL.value,
            "active_agent": "knowledge_agent",
            "agent_response": cached_response,
            "total_tokens": state.total_tokens or 0
        }

    result = await knowledge_agent.run(
        user_message=message,
        intent=intent,
        user_context=user_context.get("context_string", "")
    )
    response_text = result.output.get("response", "")
    print(f"[KNOWLEDGE] Agent returned: '{response_text[:100]}...' (grounded: {result.output.get('grounded')})")

    if response_text:
        await redis_cache.set(intent=intent, query=message, response=response_text)

    return {
        "current_state": ConversationStateEnum.KNOWLEDGE_RETRIEVAL.value,
        "active_agent": "knowledge_agent",
        "agent_response": response_text,
        "total_tokens": (state.total_tokens or 0) + result.tokens_used
    }


async def action_execution_node(state: NexusState) -> dict:
    message = state.current_message or ""
    analyzed = state.analyzed_input
    user_context = state.user_context or {}

    # In our implementation, we used dicts previously. Now it's a Pydantic object
    intent = analyzed.primary_intent if analyzed else "unclear"
    # Button bypass logic check (we would need to add intent_override if we want to keep it, but it's not in the Pydantic schema)
    # We will just pass the entities dict
    entities = analyzed.entities.model_dump() if (analyzed and analyzed.entities) else {}

    result = await action_agent.run(
        intent=intent,
        entities=entities,
        user_id=state.user_id or "anonymous",
        conversation_id=state.conversation_id or "unknown",
        user_message=message,
        user_context=user_context.get("context_string", "")
    )
    return {
        "current_state": ConversationStateEnum.ACTION_EXECUTION.value,
        "active_agent": "action_agent",
        "agent_response": result.output.get("response", ""),
        "tools_called": result.output.get("tools_called", []),
        "tool_results": result.output.get("tool_results", []),
        "total_tokens": (state.total_tokens or 0) + result.tokens_used
    }


async def quality_check_node(state: NexusState) -> dict:
    message = state.current_message or ""
    agent_response = state.agent_response or ""
    analyzed = state.analyzed_input
    intent = analyzed.primary_intent if analyzed else "unknown"
    tool_results = state.tool_results or []

    evaluation = await quality_judge.evaluate(
        user_message=message,
        agent_response=agent_response,
        tool_results=tool_results,
        intent=intent
    )

    return {
        "current_state": ConversationStateEnum.QUALITY_CHECK.value,
        "quality_scores": {
            "factual_accuracy": evaluation.factual_accuracy,
            "helpfulness": evaluation.helpfulness,
            "policy_compliance": evaluation.policy_compliance,
            "tool_correctness": evaluation.tool_correctness,
            "conversation_flow": evaluation.conversation_flow
        },
        "quality_passed": evaluation.overall_pass,
        "revision_suggestion": evaluation.revision_suggestion
    }


async def revision_node(state: NexusState) -> dict:
    from infrastructure.llm.nim_client import nim_client

    message = state.current_message or ""
    original_response = state.agent_response or ""
    feedback = state.revision_suggestion or "Please improve the quality of your response."
    tool_results = state.tool_results or []
    
    system_prompt = (
        "You are an AI assistant revising a customer support response. "
        "The original response failed our quality/compliance check. "
        "Revise the response according to the Quality Judge's specific feedback. "
        "Keep the tone professional and empathetic."
    )
    
    user_prompt = (
        f"Customer Message: {message}\n"
        f"Tool Data Available: {json.dumps(tool_results)}\n\n"
        f"Original Failed Response: {original_response}\n"
        f"Judge's Feedback: {feedback}\n\n"
        "Please provide the revised response."
    )

    result = await nim_client.complete(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        tier="standard"
    )

    new_response = result.get("content")
    if not new_response:
        new_response = original_response

    return {
        "current_state": ConversationStateEnum.REVISION.value,
        "agent_response": new_response,
        "revision_count": (state.revision_count or 0) + 1,
        "quality_passed": None, 
        "revision_suggestion": None
    }

async def escalation_node(state: NexusState) -> dict:
    analyzed = state.analyzed_input
    risk_flags = analyzed.risk_flags if analyzed else []
    intent = analyzed.primary_intent if analyzed else "unknown"
    
    sentiment_score = analyzed.sentiment.score if (analyzed and analyzed.sentiment) else 0
    
    if "suicide_risk" in risk_flags or "hardship" in risk_flags:
        reason = "Customer distress detected"
    elif "lawsuit_threat" in risk_flags:
        reason = "Legal/Lawsuit threat detected"
    elif "explicit_human_request" in risk_flags:
        reason = "Customer requested human agent"
    elif sentiment_score < -0.8:
        reason = f"Extremely negative sentiment detected ({sentiment_score})"
    else:
        reason = f"Quality check failure on intent: {intent}"
    
    handoff_message = (
        "I understand this is important to you. I'm connecting you with a specialist "
        "who can help further. Please hold while I transfer you. "
        f"Reference: {state.conversation_id or 'N/A'}"
    )
    
    print(f"[ESCALATION] Reason: {reason} | User: {state.user_id} | Conv: {state.conversation_id}")
    
    return {
        "current_state": ConversationStateEnum.ESCALATION.value,
        "escalated": True,
        "escalation_reason": reason,
        "agent_response": handoff_message,
        "handoff_package": {
            "user_id": state.user_id,
            "conversation_id": state.conversation_id,
            "escalation_reason": reason,
            "intent": intent,
            "risk_flags": risk_flags
        }
    }