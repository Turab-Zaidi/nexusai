# core/orchestrator/state_machine.py

from langgraph.graph import StateGraph, END
import os
import json
import dataclasses

from .state import NexusState, ConversationStateEnum
from core.agents.input_classifier import InputClassifier
from core.agents.knowledge_agent import KnowledgeAgent
from core.agents.action_agent import ActionAgent
from core.agents.resolution_agent import ResolutionAgent
from core.agents.escalation_agent import EscalationAgent
from core.quality.judge import QualityJudge
from core.memory.episodic_memory import episodic_memory
from infrastructure.cache.redis_cache import redis_cache


def create_graph():
    """
    Create the LangGraph state machine.
    This defines every possible state and transition.
    """

    graph = StateGraph(NexusState)

    # ── Add all nodes (states) ──────────────────────────

    graph.add_node("greeting", greeting_node)
    graph.add_node("intent_classification", intent_classification_node)
    graph.add_node("collecting_info", collecting_info_node)
    graph.add_node("knowledge_retrieval", knowledge_retrieval_node)
    graph.add_node("action_execution", action_execution_node)
    graph.add_node("resolution_execution", resolution_execution_node)
    graph.add_node("quality_check", quality_check_node)
    graph.add_node("revision", revision_node)
    graph.add_node("response_delivery", response_delivery_node)
    graph.add_node("escalation", escalation_node)

    # ── Set entry point ─────────────────────────────────

    graph.set_entry_point("greeting")

    # ── Add edges (transitions) ─────────────────────────

    graph.add_edge("greeting", "intent_classification")

    # From intent_classification → route based on AI analysis
    graph.add_conditional_edges(
        "intent_classification",
        route_after_classification,
        {
            "collecting_info": "collecting_info",
            "knowledge_retrieval": "knowledge_retrieval",
            "action_execution": "action_execution",
            "resolution_execution": "resolution_execution",
            "escalation": "escalation"
        }
    )

    graph.add_edge("collecting_info", "intent_classification")
    graph.add_edge("knowledge_retrieval", "quality_check")
    graph.add_edge("action_execution", "quality_check")
    graph.add_edge("resolution_execution", "quality_check")

    # From quality_check → route based on AI judge scores
    graph.add_conditional_edges(
        "quality_check",
        route_after_quality_check,
        {
            "response_delivery": "response_delivery",
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

    graph.add_edge("response_delivery", END)
    graph.add_edge("escalation", END)

    return graph.compile()


# ── Routing functions ────────────────────────────────────

def route_after_classification(state: NexusState) -> str:
    """Decide where to go after analyzing intent"""

    analyzed = state.get("analyzed_input", {})

    sentiment_score = analyzed.get("sentiment", {}).get("score", 0)

    if sentiment_score < -0.8:
        return "escalation"

    risk_flags = analyzed.get("risk_flags", [])
    if "legal_language" in risk_flags or "explicit_human_request" in risk_flags:
        return "escalation"

    intent = analyzed.get("primary_intent", "unclear")

    # Intents that need action tools (e.g., checking a database)
    action_intents = [
        "get_refund", "cancel_order", "change_order",
        "track_order", "check_invoice", "get_invoice",
        "payment_issue", "track_refund"
    ]

    # Intents satisfied by RAG knowledge retrieval
    knowledge_intents = [
        "check_refund_policy", "delivery_options",
        "delivery_period", "check_payment_methods",
        "check_cancellation_fee", "review", "complaint"
    ]

    # Check if we need more information before taking action
    if intent in action_intents:
        if analyzed.get("complexity") == 3:
            return "resolution_execution"
            
        entities = analyzed.get("entities", {})
        if not entities.get("order_id") and intent in ["get_refund", "cancel_order", "track_order"]:
            return "collecting_info"
        return "action_execution"

    if intent in knowledge_intents:
        return "knowledge_retrieval"

    # Default to knowledge retrieval for unclear inputs
    return "knowledge_retrieval"


def route_after_quality_check(state: NexusState) -> str:
    """Decide what to do based on quality scores"""

    if state.get("quality_passed"):
        return "response_delivery"

    revision_count = state.get("revision_count", 0)
    scores = state.get("quality_scores", {})

    # Policy violation → immediate escalation
    if scores.get("policy_compliance", 5) < 4:
        return "escalation"

    # Two failed revisions → escalate
    if revision_count >= 2:
        return "escalation"

    return "revision"


def route_after_revision(state: NexusState) -> str:
    """After revision, try quality check again"""

    revision_count = state.get("revision_count", 0)
    if revision_count >= 2:
        return "escalation"
    return "quality_check"




# Instantiate agents
input_classifier = InputClassifier("input_classifier", "fast")
knowledge_agent = KnowledgeAgent()
action_agent = ActionAgent()
resolution_agent = ResolutionAgent()
escalation_agent = EscalationAgent()
quality_judge = QualityJudge()


# ── Node Implementations ─────────────────────────────────

async def greeting_node(state: NexusState) -> dict:
    """
    Entry point. Retrieves user memories from Mem0
    to provide personalized context to downstream agents.
    """
    user_id = state.get("user_id", "anonymous")
    message = state.get("current_message", "")

    # Retrieve episodic memories for this user
    memories = await episodic_memory.retrieve(user_id=user_id, query=message)
    user_context_str = episodic_memory.format_memories_for_prompt(memories)

    return {
        "current_state": ConversationStateEnum.GREETING,
        "turn_count": state.get("turn_count", 0) + 1,
        "user_context": {"memories": memories, "context_string": user_context_str}
    }


async def intent_classification_node(state: NexusState) -> dict:
    message = state.get("current_message", "")
    result = await input_classifier.run(message)
    # Convert AnalyzedInput dataclass to dict
    analyzed_dict = dataclasses.asdict(result.output)
    return {
        "current_state": ConversationStateEnum.INTENT_CLASSIFICATION,
        "analyzed_input": analyzed_dict
    }


async def collecting_info_node(state: NexusState) -> dict:
    """
    Dynamically asks for missing information using the LLM
    instead of a hardcoded string.
    """
    from infrastructure.llm.nim_client import nim_client

    message = state.get("current_message", "")
    analyzed = state.get("analyzed_input", {})
    intent = analyzed.get("primary_intent", "unclear")
    entities = analyzed.get("entities", {})

    # Figure out what's missing
    missing_fields = []
    if not entities.get("order_id") and intent in ["get_refund", "cancel_order", "track_order", "track_refund"]:
        missing_fields.append("order number")
    if not entities.get("product_name") and intent in ["change_order"]:
        missing_fields.append("product name")

    if not missing_fields:
        missing_fields = ["more details about your request"]

    # Use LLM to generate a natural follow-up question
    response = await nim_client.complete(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a friendly customer support agent. "
                    "The customer wants help but is missing some information. "
                    "Ask them politely for the missing information in one short sentence. "
                    "Be specific about what you need."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Customer said: '{message}'\n"
                    f"Their intent is: {intent}\n"
                    f"Missing information: {', '.join(missing_fields)}\n"
                    "Generate a follow-up question asking for the missing info."
                )
            }
        ],
        tier="fast",
        max_tokens=100
    )

    follow_up = response.get("content") or f"Could you please provide your {', '.join(missing_fields)}?"

    return {
        "current_state": ConversationStateEnum.COLLECTING_INFO,
        "agent_response": follow_up
    }


async def knowledge_retrieval_node(state: NexusState) -> dict:
    message = state.get("current_message", "")
    analyzed = state.get("analyzed_input", {})
    intent = analyzed.get("primary_intent", "unclear")

    # Check Redis cache first
    cached_response = await redis_cache.get(intent=intent, query=message)
    if cached_response:
        return {
            "current_state": ConversationStateEnum.KNOWLEDGE_RETRIEVAL,
            "active_agent": "knowledge_agent",
            "agent_response": cached_response,
            "total_tokens": state.get("total_tokens", 0)  # No new tokens used
        }

    result = await knowledge_agent.run(user_message=message, intent=intent)
    response_text = result.output.get("response", "")

    # Cache the response for future similar queries
    if response_text:
        await redis_cache.set(intent=intent, query=message, response=response_text)

    return {
        "current_state": ConversationStateEnum.KNOWLEDGE_RETRIEVAL,
        "active_agent": "knowledge_agent",
        "agent_response": response_text,
        "total_tokens": state.get("total_tokens", 0) + result.tokens_used
    }


async def action_execution_node(state: NexusState) -> dict:
    message = state.get("current_message", "")
    analyzed = state.get("analyzed_input", {})
    intent = analyzed.get("primary_intent", "unclear")
    entities = analyzed.get("entities", {})
    
    result = await action_agent.run(
        intent=intent,
        entities=entities,
        user_id=state.get("user_id", "anonymous"),
        conversation_id=state.get("conversation_id", "unknown"),
        user_message=message
    )
    return {
        "current_state": ConversationStateEnum.ACTION_EXECUTION,
        "active_agent": "action_agent",
        "agent_response": result.output.get("response", ""),
        "tools_called": result.output.get("tools_called", []),
        "tool_results": result.output.get("tool_results", []),
        "total_tokens": state.get("total_tokens", 0) + result.tokens_used
    }


async def resolution_execution_node(state: NexusState) -> dict:
    message = state.get("current_message", "")
    analyzed = state.get("analyzed_input", {})
    intent = analyzed.get("primary_intent", "unclear")
    entities = analyzed.get("entities", {})
    
    result = await resolution_agent.run(
        intent=intent,
        entities=entities,
        user_id=state.get("user_id", "anonymous"),
        conversation_id=state.get("conversation_id", "unknown"),
        user_message=message
    )
    return {
        "current_state": ConversationStateEnum.RESOLUTION_EXECUTION,
        "active_agent": "resolution_agent",
        "agent_response": result.output.get("response", ""),
        "tools_called": result.output.get("tools_called", []),
        "tool_results": result.output.get("tool_results", []),
        "total_tokens": state.get("total_tokens", 0) + result.tokens_used
    }


async def quality_check_node(state: NexusState) -> dict:
    message = state.get("current_message", "")
    agent_response = state.get("agent_response", "")
    tool_results = state.get("tool_results", [])
    analyzed = state.get("analyzed_input", {})
    intent = analyzed.get("primary_intent", "unclear")

    result = await quality_judge.evaluate(
        user_message=message,
        agent_response=agent_response,
        tool_results=tool_results,
        intent=intent
    )
    
    scores = {
        "factual_accuracy": result.factual_accuracy,
        "helpfulness": result.helpfulness,
        "policy_compliance": result.policy_compliance,
        "tool_correctness": result.tool_correctness,
        "conversation_flow": result.conversation_flow
    }

    return {
        "current_state": ConversationStateEnum.QUALITY_CHECK,
        "quality_passed": result.overall_pass,
        "quality_scores": scores,
        "revision_suggestion": result.revision_suggestion
    }


async def revision_node(state: NexusState) -> dict:
    """
    Real revision logic: takes the judge's feedback and asks the
    original agent to regenerate a better response.
    """
    from infrastructure.llm.nim_client import nim_client

    message = state.get("current_message", "")
    original_response = state.get("agent_response", "")
    scores = state.get("quality_scores", {})
    suggestion = state.get("revision_suggestion", "")
    tool_results = state.get("tool_results", [])
    active_agent = state.get("active_agent", "unknown")

    # Build feedback context for the rewrite
    failing_dims = []
    for dim, score in scores.items():
        threshold = {"factual_accuracy": 4, "helpfulness": 3, "policy_compliance": 4,
                      "tool_correctness": 4, "conversation_flow": 3}.get(dim, 3)
        if score < threshold:
            failing_dims.append(f"{dim}: {score}/5 (needs {threshold}+)")

    feedback_text = "\n".join(failing_dims)

    # Ask the LLM to rewrite the response incorporating the feedback
    revision_response = await nim_client.complete(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a customer support agent. Your previous response was rejected "
                    "by the quality evaluator. You must rewrite it to fix the identified issues.\n\n"
                    "RULES:\n"
                    "- Fix ONLY the issues identified below\n"
                    "- Keep the factual content from tool results accurate\n"
                    "- Be empathetic and helpful\n"
                    "- Keep the response under 100 words"
                )
            },
            {
                "role": "user",
                "content": (
                    f"Customer message: '{message}'\n\n"
                    f"Your previous response:\n{original_response}\n\n"
                    f"Quality scores that FAILED:\n{feedback_text}\n\n"
                    f"Judge suggestion: {suggestion}\n\n"
                    f"Tool results for reference:\n{json.dumps(tool_results, indent=2)}\n\n"
                    "Write an improved response that fixes these issues:"
                )
            }
        ],
        tier="standard",
        max_tokens=300
    )

    revised_text = revision_response.get("content", original_response)

    return {
        "current_state": ConversationStateEnum.REVISION,
        "revision_count": state.get("revision_count", 0) + 1,
        "agent_response": revised_text,
        "total_tokens": state.get("total_tokens", 0) + revision_response.get("prompt_tokens", 0) + revision_response.get("completion_tokens", 0)
    }


async def response_delivery_node(state: NexusState) -> dict:
    """
    Final delivery. Stores memory about this conversation
    for future personalization.
    """
    user_id = state.get("user_id", "anonymous")
    message = state.get("current_message", "")
    response = state.get("agent_response", "")
    analyzed = state.get("analyzed_input", {})
    intent = analyzed.get("primary_intent", "unknown")

    # Store episodic memory about this conversation
    conversation_text = (
        f"Customer asked: {message}\n"
        f"Intent: {intent}\n"
        f"AI response: {response}"
    )
    await episodic_memory.store(
        user_id=user_id,
        conversation_text=conversation_text,
        metadata={"intent": intent, "resolved": True}
    )

    return {"current_state": ConversationStateEnum.RESPONSE_DELIVERY}


async def escalation_node(state: NexusState) -> dict:
    result = await escalation_agent.run(state)
    return {
        "current_state": ConversationStateEnum.ESCALATION,
        "escalated": True,
        "escalation_reason": result.output.get("handoff_package", {}).get("escalation_reason", "unknown"),
        "agent_response": result.output.get("response", ""),
        "handoff_package": result.output.get("handoff_package", {})
    }