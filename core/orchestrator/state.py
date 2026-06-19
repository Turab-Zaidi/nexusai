# core/orchestrator/state.py

from typing import Optional, Any, Annotated
from enum import Enum
from pydantic import BaseModel, Field
import operator

from langgraph.graph.message import add_messages
from core.agents.input_classifier import AnalyzedInput


class ConversationStateEnum(str, Enum):
    STARTUP = "STARTUP"
    CONTEXT_LOADER = "CONTEXT_LOADER"
    INTENT_CLASSIFICATION = "INTENT_CLASSIFICATION"
    COLLECTING_INFO = "COLLECTING_INFO"
    KNOWLEDGE_RETRIEVAL = "KNOWLEDGE_RETRIEVAL"
    ACTION_EXECUTION = "ACTION_EXECUTION"
    QUALITY_CHECK = "QUALITY_CHECK"
    REVISION = "REVISION"
    ESCALATION = "ESCALATION"
    END = "END"


class NexusState(BaseModel):
    # Identifiers
    conversation_id: str = ""
    user_id: str = ""
    session_id: str = ""
    channel: str = "web"

    # LangGraph Reducer for messages
    messages: Annotated[list, add_messages] = Field(default_factory=list)

    current_message: str = ""
    current_state: str = ""
    analyzed_input: Optional[AnalyzedInput] = None

    # Agent work
    active_agent: Optional[str] = None
    agent_response: Optional[str] = None
    tools_called: list = Field(default_factory=list)
    tool_results: list = Field(default_factory=list)

    # Quality
    quality_scores: Optional[dict] = None
    revision_count: int = 0
    quality_passed: Optional[bool] = None
    revision_suggestion: Optional[str] = None

    # Memory and context
    user_context: Optional[dict] = None
    conversation_summary: str = ""

    # Routing
    routing_decision: Optional[str] = None

    # Escalation
    escalated: bool = False
    escalation_reason: Optional[str] = None
    handoff_package: Optional[dict] = None

    # Tracking
    turn_count: int = 0
    total_tokens: int = 0