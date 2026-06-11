# core/agents/resolution_agent.py
import json
from .base_agent import BaseAgent, AgentResult
from langfuse.decorators import observe
from core.tools.implementations.order_lookup import lookup_order
from core.tools.implementations.refund_processor import process_refund
from core.tools.implementations.ticket_creator import create_ticket
from core.tools.implementations.user_profile import get_user_profile

class ResolutionAgent(BaseAgent):
    """
    Handles complex multi-step workflows.
    Generates a plan using 70B, executes tools sequentially, and synthesizes a final response.
    """
    def __init__(self):
        super().__init__(
            name="resolution_agent",
            model_tier="advanced" # Uses Llama 3.1 70B for complex reasoning
        )

    @observe(as_type="span", name="resolution_agent")
    async def run(
        self,
        intent: str,
        entities: dict,
        user_id: str,
        conversation_id: str,
        user_message: str
    ) -> AgentResult:
        
        tools_called = []
        tool_results = []
        
        # ── Step 1: Generate Execution Plan ────────────────────────────
        plan_system_prompt = (
            "You are a Senior Customer Resolution Manager.\n"
            "You must create a step-by-step plan to resolve the customer's complex issue.\n"
            "Available tools:\n"
            "1. lookup_order(order_id: str)\n"
            "2. get_user_profile(user_id: str)\n"
            "3. process_refund(order_id: str, amount: float, reason: str)\n"
            "4. create_ticket(user_id: str, issue_description: str, priority: str)\n"
            "\n"
            "Respond ONLY with a valid JSON object in this format:\n"
            "{\n"
            '  "steps": [\n'
            '    {"tool": "tool_name", "args": {"arg1": "value"}}\n'
            "  ]\n"
            "}"
        )
        
        plan_user_message = (
            f"User ID: {user_id}\n"
            f"Intent: {intent}\n"
            f"Extracted Entities: {json.dumps(entities)}\n"
            f"Customer Message: '{user_message}'\n\n"
            "Generate the execution plan."
        )
        
        plan_response = await self.call_llm_json(
            system_prompt=plan_system_prompt,
            user_message=plan_user_message
        )
        
        # ── Step 2: Execute Plan Sequentially ──────────────────────────
        plan_data = plan_response.get("parsed") or {}
        steps = plan_data.get("steps", [])
        
        for step in steps:
            tool_name = step.get("tool")
            args = step.get("args", {})
            
            result = None
            if tool_name == "lookup_order":
                order_id = args.get("order_id")
                if order_id:
                    result = await lookup_order(order_id)
            elif tool_name == "get_user_profile":
                uid = args.get("user_id") or user_id
                result = await get_user_profile(uid)
            elif tool_name == "process_refund":
                order_id = args.get("order_id")
                amount = args.get("amount", 0.0)
                reason = args.get("reason", "Customer request")
                if order_id and amount:
                    # In a real app we'd convert string to float, safe cast here:
                    try:
                        amount = float(amount)
                        result = await process_refund(order_id, amount, reason)
                    except ValueError:
                        result = {"ok": False, "error": "Invalid amount format"}
            elif tool_name == "create_ticket":
                uid = args.get("user_id") or user_id
                desc = args.get("issue_description", "Escalated issue")
                priority = args.get("priority", "high")
                result = await create_ticket(uid, desc, priority)
                
            if result:
                tools_called.append({"tool_name": tool_name, "input": args})
                tool_results.append({tool_name: result})
                
                # Halt if a critical tool failed to prevent cascading errors
                if not result.get("ok", True):
                    break

        # ── Step 3: Synthesize Final Response ──────────────────────────
        final_system_prompt = (
            "You are a Senior Customer Resolution Manager. "
            "Review the customer's issue and the results of the tools you called to resolve it. "
            "Write a clear, empathetic, and definitive final response to the customer. "
            "Explain exactly what actions were taken (e.g., refund processed, ticket created). "
            "Keep it under 150 words."
        )
        
        final_user_message = (
            f"Customer Message: '{user_message}'\n"
            f"Tool Execution Results:\n{json.dumps(tool_results, indent=2)}\n"
        )
        
        final_response = await self.call_llm(
            system_prompt=final_system_prompt,
            user_message=final_user_message
        )
        
        # Combine token usage from both LLM calls
        final_response["prompt_tokens"] = final_response.get("prompt_tokens", 0) + plan_response.get("prompt_tokens", 0)
        final_response["completion_tokens"] = final_response.get("completion_tokens", 0) + plan_response.get("completion_tokens", 0)
        final_response["latency_ms"] = final_response.get("latency_ms", 0) + plan_response.get("latency_ms", 0)
        
        return self.make_result(
            success=True,
            output={
                "response": final_response["content"],
                "tools_called": tools_called,
                "tool_results": tool_results
            },
            llm_response=final_response
        )
