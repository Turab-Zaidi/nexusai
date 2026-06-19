import json
import logging
from .base_agent import BaseAgent, AgentResult
from core.tools.implementations.fintech_tools import (
    get_user_profile,
    get_recent_transactions,
    freeze_card,
    unfreeze_card,
    report_stolen_card,
    submit_dispute,
    waive_fee,
    analyze_spending,
    search_transactions
)


class ActionAgent(BaseAgent):
    """
    Executes FinTech actions against the SQLite database.
    Handles: freeze_card, unfreeze_card, submit_dispute, fee_waiver,
             check_transaction, report_fraud, request_virtual_card.
    """

    def __init__(self):
        super().__init__(name="action_agent", model_tier="standard")

    async def run(
        self,
        intent: str,
        entities: dict,
        user_id: str,
        conversation_id: str,
        user_message: str,
        user_context: str = ""
    ) -> AgentResult:

        tools_called = []
        tool_results = []

        # ── FREEZE / UNFREEZE CARD ─────────────────────────────────────────
        if intent in ["freeze_card", "unfreeze_card", "report_fraud"]:
            card_id = entities.get("card_id")
            
            if intent == "freeze_card":
                result = await freeze_card(card_id=card_id, user_id=user_id)
                tool_name = "freeze_card"
                context_success = "Card has been frozen successfully."
                context_error = f"Could not freeze card: {result['error']}"
            elif intent == "unfreeze_card":
                result = await unfreeze_card(card_id=card_id, user_id=user_id)
                tool_name = "unfreeze_card"
                context_success = "Card has been unfrozen successfully."
                context_error = f"Could not unfreeze card: {result['error']}"
            else: # report_fraud
                result = await report_stolen_card(card_id=card_id, user_id=user_id)
                tool_name = "report_stolen_card"
                context_success = "Card has been permanently blocked and reported stolen."
                context_error = f"Could not report card: {result['error']}"

            tools_called.append({"tool_name": tool_name, "input": {"card_id": card_id, "user_id": user_id}})
            tool_results.append(result)
            context = context_success if result["ok"] else context_error
            llm_response = await self._generate_response(user_message, tool_results, user_context, context)
            return self.make_result(result["ok"], {"response": llm_response["content"], "tools_called": tools_called, "tool_results": tool_results}, llm_response)

        # ── SUBMIT DISPUTE ─────────────────────────────────────────────────
        if intent == "submit_dispute":
            transaction_id = entities.get("transaction_id")
            
            result = await submit_dispute(
                transaction_id=transaction_id,
                user_id=user_id,
                reason=entities.get("reason", "Customer-initiated dispute")
            )
            tools_called.append({"tool_name": "submit_dispute", "input": {"transaction_id": transaction_id}})
            tool_results.append(result)
            context = f"Dispute submitted. Reference: {result['data'].get('reference')}" if result["ok"] else f"Dispute failed: {result['error']}"
            llm_response = await self._generate_response(user_message, tool_results, user_context, context)
            return self.make_result(result["ok"], {"response": llm_response.get("content") or "Error: Blank response from AI", "tools_called": tools_called, "tool_results": tool_results}, llm_response)

        # ── FEE WAIVER ────────────────────────────────────────────────────
        if intent == "fee_waiver":
            transaction_id = entities.get("transaction_id")
            
            result = await waive_fee(transaction_id=transaction_id, user_id=user_id)
            tools_called.append({"tool_name": "waive_fee", "input": {"transaction_id": transaction_id, "user_id": user_id}})
            tool_results.append(result)
            context = f"Fee of ${result['data'].get('fee_amount')} waived. Reference: {result['data'].get('reference')}" if result["ok"] else f"Fee waiver failed: {result['error']}"
            llm_response = await self._generate_response(user_message, tool_results, user_context, context)
            return self.make_result(result["ok"], {"response": llm_response["content"], "tools_called": tools_called, "tool_results": tool_results}, llm_response)

        # ── CHECK TRANSACTION / ACCOUNT ───────────────────────────────────
        if intent in ["check_transaction", "account_info"]:
            profile = await get_user_profile(user_id)
            
            merchant = entities.get("merchant_name")
            amount = entities.get("amount")
            
            if intent == "check_transaction" and (merchant or amount):
                txns = await search_transactions(user_id, merchant_name=merchant, amount=amount)
                tool_name = "search_transactions"
                tool_input = {"user_id": user_id, "merchant_name": merchant, "amount": amount}
            else:
                txns = await get_recent_transactions(user_id, limit=5)
                tool_name = "get_recent_transactions"
                tool_input = {"user_id": user_id}
                
            tools_called.extend([
                {"tool_name": "get_user_profile", "input": {"user_id": user_id}},
                {"tool_name": tool_name, "input": tool_input}
            ])
            tool_results.extend([profile, txns])
            llm_response = await self._generate_response(user_message, tool_results, user_context)
            return self.make_result(True, {"response": llm_response["content"], "tools_called": tools_called, "tool_results": tool_results}, llm_response)

        # ── FINANCIAL ANALYSIS ────────────────────────────────────────────
        if intent == "financial_analysis":
            days = entities.get("time_period_days") or 30
            category = entities.get("expense_category")
            
            result = await analyze_spending(user_id=user_id, days=days, category=category)
            tools_called.append({"tool_name": "analyze_spending", "input": {"user_id": user_id, "days": days, "category": category}})
            tool_results.append(result)
            
            context = "Financial analysis complete."
            llm_response = await self._generate_response(user_message, tool_results, user_context, context)
            return self.make_result(result["ok"], {"response": llm_response["content"], "tools_called": tools_called, "tool_results": tool_results}, llm_response)

        # ── UNHANDLED INTENT ───────────────────────────────────────────────
        return self.make_result(
            False,
            {"response": "I'll connect you with a specialist for that request.", "tools_called": [], "tool_results": []},
            error="Intent not handled by action agent"
        )

    async def _generate_response(
        self,
        user_message: str,
        tool_results: list,
        user_context: str = "",
        context: str = None
    ) -> dict:
        context_block = f"\nACTION RESULT: {context}" if context else ""
        history_block = f"\nCUSTOMER HISTORY:\n{user_context}" if user_context else ""

        result = await self.call_llm(
            system_prompt=(
                "You are the Action Response Generator for Nexus Bank. "
                "Your job is to read the raw JSON 'Tool Results' that the backend has already executed on the user's behalf, "
                "and translate those results into a professional, empathetic response to the customer.\n\n"
                "UNDERSTANDING TOOL RESULTS:\n"
                "- If 'ok': true, the action (e.g., freezing a card, waiving a fee) succeeded. Confirm this to the user and provide any reference numbers.\n"
                "- If 'ok': false, the action failed (e.g., card already frozen). Politely explain the error.\n"
                "- For data tools (like financial analysis or checking transactions), summarize the sums, counts, and dates naturally.\n\n"
                "CRITICAL RULES:\n"
                "1. NEVER promise or guarantee outcomes that are not explicitly confirmed in the Tool Results.\n"
                "2. Be specific. Always quote exact amounts, the last 4 digits of cards, and reference numbers found in the JSON.\n"
                "3. NO FINANCIAL ADVICE. When summarizing financial analysis, only provide objective facts (e.g., 'You spent $400 on dining'). Do NOT provide subjective advice (e.g., 'You should cut back on dining').\n"
                "4. Keep the response under 80 words. Be concise and professional."
            ),
            user_message=(
                f"Customer message: '{user_message}'\n"
                f"{history_block}"
                f"\nTool Results:\n{json.dumps(tool_results, indent=2)}"
                f"{context_block}"
            )
        )
        logging.getLogger(__name__).error(f"[ACTION AGENT] Generated response: {result.get('content')}")
        return result