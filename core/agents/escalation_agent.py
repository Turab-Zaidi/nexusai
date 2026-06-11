# core/agents/escalation_agent.py
import json
from typing import Dict, Any
from .base_agent import BaseAgent, AgentResult
from langfuse.decorators import observe

class EscalationAgent(BaseAgent):
    """
    Handles graceful handoffs to human agents.
    Assembles context, generates summaries, and recommends resolutions.
    """
    def __init__(self):
        super().__init__(
            name="escalation_agent",
            model_tier="fast" # Uses Llama 3.1 8B for fast summarization
        )

    @observe(as_type="span", name="escalation_agent")
    async def run(self, state: Dict[str, Any]) -> AgentResult:
        # 1. Gather all context from the state
        user_message = state.get("current_message", "")
        analyzed = state.get("analyzed_input", {})
        intent = analyzed.get("primary_intent", "unclear")
        sentiment = analyzed.get("sentiment", {}).get("label", "neutral")
        sentiment_score = analyzed.get("sentiment", {}).get("score", 0.0)
        risk_flags = analyzed.get("risk_flags", [])
        tools_called = state.get("tools_called", [])
        tool_results = state.get("tool_results", [])
        
        # 2. Infer Escalation Reason (since router only routes)
        escalation_reason = state.get("escalation_reason")
        if not escalation_reason:
            if sentiment_score < -0.8:
                escalation_reason = "high_negative_sentiment"
            elif "legal_language" in risk_flags:
                escalation_reason = "legal_risk"
            elif "explicit_human_request" in risk_flags:
                escalation_reason = "explicit_human_request"
            elif state.get("quality_scores", {}).get("policy_compliance", 5) < 4:
                escalation_reason = "policy_violation"
            elif state.get("revision_count", 0) >= 2:
                escalation_reason = "quality_revision_limit_exceeded"
            else:
                escalation_reason = "unknown_trigger"
        
        # 3. Determine Priority Level
        priority = "medium"
        if "legal_language" in risk_flags or sentiment_score < -0.8:
            priority = "critical"
        elif "explicit_human_request" in risk_flags or escalation_reason == "policy_violation":
            priority = "high"
            
        # 4. Generate Summary & Recommendation using Llama 8B
        system_prompt = (
            "You are a Customer Handoff Specialist.\n"
            "An AI agent failed to resolve a customer issue, and it is being escalated to a human.\n"
            "Generate a clear, concise JSON handoff package.\n"
            "Format:\n"
            "{\n"
            '  "situation_summary": "1-2 sentences explaining what the customer wants",\n'
            '  "actions_attempted": "1-2 sentences explaining what the AI already tried",\n'
            '  "recommended_resolution": "What the human agent should do next"\n'
            "}"
        )
        
        user_prompt = (
            f"Customer Message: '{user_message}'\n"
            f"Intent Identified: {intent}\n"
            f"Escalation Reason: {escalation_reason}\n"
            f"Tools Used by AI: {json.dumps(tools_called)}\n"
            f"Tool Results: {json.dumps(tool_results)}\n"
        )
        
        llm_response = await self.call_llm_json(
            system_prompt=system_prompt,
            user_message=user_prompt
        )
        
        parsed_response = llm_response.get("parsed") or {}
        
        # 5. Assemble final package
        handoff_package = {
            "escalation_reason": escalation_reason,
            "priority": priority,
            "customer_sentiment": sentiment,
            "situation_summary": parsed_response.get("situation_summary", "Failed to generate summary"),
            "actions_attempted": parsed_response.get("actions_attempted", "None"),
            "recommended_resolution": parsed_response.get("recommended_resolution", "Review manually"),
            "tools_history": tools_called
        }
        
        # 6. Generate user-facing message
        user_facing_response = (
            "I apologize, but I need to escalate this to a human agent who can assist you further. "
            f"I have already summarized your issue and forwarded it to our team with a priority of '{priority}'. "
            "Someone will be with you shortly."
        )
        
        return self.make_result(
            success=True,
            output={
                "response": user_facing_response,
                "handoff_package": handoff_package
            },
            llm_response=llm_response
        )
