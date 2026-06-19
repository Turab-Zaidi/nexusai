# core/orchestrator/pipeline.py

import uuid
from typing import Dict, Any
from .state_machine import create_graph
from langfuse.decorators import observe, langfuse_context
from langgraph.types import Command
from core.agents.input_classifier import AnalyzedInput, EntityResult

# We compile the graph once when this module is loaded
app = create_graph()

class Pipeline:
    @observe()
    async def process_stream(self, message: str, session_id: str, user_id: str, conversation_history: list = None, intent_override: str = None, transaction_id: str = None):
        """
        Yields state updates from LangGraph as they happen.
        Handles both fresh executions and resuming interrupted graphs.
        """
        langfuse_context.update_current_trace(
            name="nexusai_conversation_stream",
            session_id=session_id,
            user_id=user_id,
            input=message
        )
        
        config = {"configurable": {"thread_id": session_id}}

        # Check if we're resuming a paused graph
        graph_state = await app.aget_state(config)

        if graph_state and graph_state.next:
            # Resume the interrupted graph with the user's reply
            async for event in app.astream(
                Command(resume=message),
                config=config,
                stream_mode="updates"
            ):
                yield event
        else:
            # Fresh turn
            initial_state = {
                "conversation_id": str(uuid.uuid4()),
                "user_id": user_id,
                "session_id": session_id,
                "channel": "web",
                "messages": conversation_history or [],
                "current_message": message,
                "current_state": "STARTUP",
                "analyzed_input": AnalyzedInput(
                    intent_override=intent_override,
                    entities=EntityResult(transaction_id=transaction_id) if transaction_id else EntityResult()
                ) if intent_override else None,
                "active_agent": None,
                "agent_response": None,
                "tools_called": [],
                "tool_results": [],
                "quality_scores": None,
                "revision_count": 0,
                "quality_passed": None,
                "revision_suggestion": None,
                "user_context": None,
                "conversation_summary": "",
                "routing_decision": None,
                "escalated": False,
                "escalation_reason": None,
                "handoff_package": None,
                "turn_count": 0,
                "total_tokens": 0
            }

            async for event in app.astream(initial_state, config=config, stream_mode="updates"):
                yield event

# Singleton instance
pipeline = Pipeline()
