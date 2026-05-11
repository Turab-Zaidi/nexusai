# core/orchestrator/state.py

from typing import Optional, Any, Annotated
from dataclasses import dataclass, field
from enum import Enum
from typing import TypedDict
import operator

from langgraph.graph.message import add_messages


class ConversationStateEnum(str, Enum):
    STARTUP = "STARTUP"
    GREETING = "GREETING"
    INTENT_CLASSIFICATION = "INTENT_CLASSIFICATION"
    COLLECTING_INFO = "COLLECTING_INFO"
    KNOWLEDGE_RETRIEVAL = "KNOWLEDGE_RETRIEVAL"
    ACTION_EXECUTION = "ACTION_EXECUTION"
    RESOLUTION_EXECUTION = "RESOLUTION_EXECUTION"
    QUALITY_CHECK = "QUALITY_CHECK"
    REVISION = "REVISION"
    RESPONSE_DELIVERY = "RESPONSE_DELIVERY"
    ESCALATION = "ESCALATION"
    END = "END"


class NexusState(TypedDict):
    # Identifiers
    conversation_id: str
    user_id: str
    session_id: str
    channel: str

    messages: Annotated[list, add_messages]

    current_message: str
    current_state: str
    analyzed_input: Optional[dict]  # AnalyzedInput from Day 5 as a dict

    # Agent work
    active_agent: Optional[str]
    agent_response: Optional[str]
    tools_called: list
    tool_results: list

    # Quality
    quality_scores: Optional[dict]
    revision_count: int
    quality_passed: Optional[bool]

    # Memory and context
    user_context: Optional[dict]
    conversation_summary: str

    # Routing
    routing_decision: Optional[str]

    # Escalation
    escalated: bool
    escalation_reason: Optional[str]
    handoff_package: Optional[dict]

    # Tracking
    turn_count: int
    total_tokens: int