# core/orchestrator/pipeline.py

import uuid
from typing import Dict, Any
from .state_machine import create_graph
from langfuse.decorators import observe, langfuse_context

# We compile the graph once when this module is loaded
app = create_graph()

class Pipeline:
    @observe()
    async def process(self, message: str, session_id: str, user_id: str) -> Dict[str, Any]:
        """
        Process a single message through the LangGraph state machine.
        """
        langfuse_context.update_current_trace(
            name="nexusai_conversation",
            session_id=session_id,
            user_id=user_id,
            input=message
        )
        
        initial_state = {
            "conversation_id": str(uuid.uuid4()),
            "user_id": user_id,
            "session_id": session_id,
            "channel": "web",
            "messages": [],
            "current_message": message,
            "current_state": "STARTUP",
            "analyzed_input": None,
            "active_agent": None,
            "agent_response": None,
            "tools_called": [],
            "tool_results": [],
            "quality_scores": None,
            "revision_count": 0,
            "quality_passed": None,
            "user_context": None,
            "conversation_summary": "",
            "routing_decision": None,
            "escalated": False,
            "escalation_reason": None,
            "handoff_package": None,
            "turn_count": 0,
            "total_tokens": 0
        }

        # Run the LangGraph state machine
        final_state = await app.ainvoke(initial_state)

        langfuse_context.update_current_trace(
            output=final_state.get("agent_response"),
            tags=[final_state.get("analyzed_input", {}).get("primary_intent", "unknown")]
        )

        # Build the final response dict from the final state
        return {
            "session_id": session_id,
            "message_received": message,
            "response": final_state.get("agent_response") or "I apologize, but I am unable to assist at this moment.",
            "escalated": final_state.get("escalated", False),
            "escalation_reason": final_state.get("escalation_reason"),
            "handoff_package": final_state.get("handoff_package"),
            "agent_used": final_state.get("active_agent"),
            "tools_called": final_state.get("tools_called", []),
            "analysis": final_state.get("analyzed_input", {}),
            "status": "complete"
        }

# Singleton instance
pipeline = Pipeline()
