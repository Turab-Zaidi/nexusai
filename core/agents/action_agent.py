# core/agents/action_agent.py
import json
from .base_agent import BaseAgent, AgentResult
from core.tools.implementations.order_lookup import lookup_order
from core.tools.implementations.refund_processor import process_refund
from core.tools.implementations.ticket_creator import create_ticket

class ActionAgent(BaseAgent):
    """
    Executes tools and generates responses based on tool results.
    Handles: order lookup, refunds, tickets.
    """

    APPROVAL_THRESHOLD = 500.0  
    def __init__(self):
        super().__init__(
            name="action_agent",
            model_tier="standard" 
        )

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
        final_response = ""
        llm_call_details = {}

        # ── Handle intents that require an order_id ────────────────────────
        if intent in ["track_order", "get_refund", "cancel_order"]:
            order_id = entities.get("order_id")
            if not order_id:
                # If the entity wasn't found, ask the user for it.
                return self._missing_entity_result("order number")

            # Call the order lookup tool
            order_result = await lookup_order(order_id)
            tools_called.append({"tool_name": "lookup_order", "input": {"order_id": order_id}})
            tool_results.append(order_result)

            if not order_result["ok"]:
                # The tool returned an error (e.g., order not found)
                llm_response = await self._generate_error_response(user_message, order_result["error"])
                return self.make_result(
                    success=True,
                    output={"response": llm_response["content"], "tools_called": tools_called, "tool_results": tool_results},
                    llm_response=llm_response
                )

            order_data = order_result["data"]

            # If the intent is a refund, proceed with refund logic
            if intent == "get_refund":
                if not order_data["refund_eligible"]:
                    context = "Order is not eligible for a refund."
                elif order_data["amount"] > self.APPROVAL_THRESHOLD:
                    context = f"Refund of ${order_data['amount']} is above the ${self.APPROVAL_THRESHOLD} threshold and requires manager approval. A ticket has been created."
                else:
                    # Process the refund directly
                    refund_result = await process_refund(order_id, order_data["amount"], "Customer request")
                    tools_called.append({"tool_name": "process_refund", "input": {"order_id": order_id, "amount": order_data["amount"]}})
                    tool_results.append(refund_result)
                    context = "Refund processed successfully." if refund_result["ok"] else f"Refund failed: {refund_result['error']}"
                
                llm_response = await self._generate_response(user_message, tool_results, context)
                return self.make_result(
                    success=True,
                    output={"response": llm_response["content"], "tools_called": tools_called, "tool_results": tool_results},
                    llm_response=llm_response
                )
            
            # For other order-related intents (track_order, etc.)
            llm_response = await self._generate_response(user_message, tool_results)
            return self.make_result(
                success=True,
                output={"response": llm_response["content"], "tools_called": tools_called, "tool_results": tool_results},
                llm_response=llm_response
            )

        llm_response = await self._generate_error_response(user_message, f"The intent '{intent}' is not handled by this agent.")
        return self.make_result(
            success=False,
            output={"response": "I'll need to connect you with our support team to handle this request.", "tools_called": [], "tool_results": []},
            llm_response=llm_response,
            error="Intent not handled by action agent"
        )


    async def _generate_response(self, user_message: str, tool_results: list, context: str = None) -> dict:
        context_info = f"Additional Context: {context}\n" if context else ""
        
        return await self.call_llm(
            system_prompt=(
                "You are a helpful customer support agent. "
                "Generate a clear, empathetic response based on the "
                "tool results provided. Be specific with details like "
                "amounts, dates, and order status. Keep the response under 100 words."
            ),
            user_message=(
                f"Customer message: '{user_message}'\n\n"
                f"Tool Results:\n{json.dumps(tool_results, indent=2)}\n"
                f"{context_info}"
            )
        )

    async def _generate_error_response(self, user_message: str, error: str) -> dict:
        return await self.call_llm(
            system_prompt=(
                "You are a helpful customer support agent. "
                "Apologize that you could not complete the request due to an error. "
                "Explain the error clearly and suggest what the user can do next (e.g., check the order number)."
            ),
            user_message=(
                f"Customer message: '{user_message}'\n"
                f"Error Encountered: '{error}'"
            )
        )

    def _missing_entity_result(self, missing_entity: str) -> AgentResult:
        # This is a special case that doesn't need an LLM call.
        response = f"I can help with that. Could you please provide the {missing_entity}?"
        return AgentResult(
            success=True,
            output={"response": response, "tools_called": [], "tool_results": []},
            agent_name=self.name,
            model_tier=self.model_tier,
            latency_ms=0,
            tokens_used=0
        )