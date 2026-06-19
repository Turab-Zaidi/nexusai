# core/agents/input_classifier.py

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional, Any, Literal

from .base_agent import BaseAgent, AgentResult
from langfuse.decorators import observe


from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class IntentResult(BaseModel):
    primary_intent: Literal[
        "fee_waiver", "freeze_card", "unfreeze_card", "submit_dispute", 
        "check_transaction", "report_fraud", "request_virtual_card", 
        "financial_analysis", "policy_question", "check_fees", "general_inquiry", 
        "check_dispute_status", "account_info", "regulation_question", 
        "complaint", "unclear"
    ] = Field(description="The primary intent of the user message.")
    secondary_intent: Optional[str] = Field(default=None)
    confidence: float = Field(ge=0.0, le=1.0)

class EntityResult(BaseModel):
    transaction_id: Optional[str] = Field(default=None)
    card_id: Optional[str] = Field(default=None)
    amount: Optional[float] = Field(default=None)
    merchant_name: Optional[str] = Field(default=None)
    time_period_days: Optional[int] = Field(default=None, description="Number of days for financial analysis (e.g., 'last month' = 30)")
    expense_category: Optional[Literal["dining", "transport", "retail", "subscription", "fee", "transfer"]] = Field(default=None, description="Category of spending")

class SentimentResult(BaseModel):
    score: float = Field(ge=-1.0, le=1.0)
    label: Literal["satisfied", "neutral", "frustrated", "angry"] = Field(description="The sentiment label.")

class RiskResult(BaseModel):
    flags: list[str] = Field(default_factory=list)
    complexity: int = Field(ge=1, le=3)
    language: str = Field(default="en")

class AnalyzedInput(BaseModel):
    raw_message: str = ""
    primary_intent: str = "unclear"
    secondary_intent: Optional[str] = None
    confidence: float = 0.5
    entities: EntityResult = Field(default_factory=EntityResult)
    sentiment: SentimentResult = Field(default_factory=lambda: SentimentResult(score=0.0, label="neutral"))
    language: str = "en"
    complexity: int = 1       
    risk_flags: list[str] = Field(default_factory=list)
    word_count: int = 0
    intent_override: Optional[str] = None


class InputClassifier(BaseAgent):
    """
    Analyzes raw customer messages.
    Runs 4 analyses in parallel using asyncio.gather.
    Uses fast tier (Llama 3.1 8B) for speed.
    """

    INTENT_SYSTEM = """You are the master routing brain for Nexus Bank's AI Support Agent. 
            Your sole job is to classify the customer's primary intent into one of our exact predefined categories.

            RULES:
            1. If the user mentions multiple issues (e.g., "I lost my card and also what are your fees?"), ALWAYS prioritize the Action Intent (e.g., freeze_card) over the Knowledge Intent. Security is paramount.
            2. If the user is just venting but asking no question, use 'complaint'.
            3. Do NOT output plain text, you must invoke the structured_output tool.

            VALID ACTION INTENTS (Requires Database Action):
            - fee_waiver: Asking to remove/waive a fee.
            - freeze_card / unfreeze_card: Temporarily locking/unlocking a card.
            - submit_dispute: Disputing a transaction.
            - check_transaction: Checking recent spending or asking about a specific charge.
            - report_fraud: Permanently blocking a card due to theft/fraud.
            - request_virtual_card: Creating a new virtual card.
            - financial_analysis: Asking about spending trends, budgeting, or categorical sums.
            - account_info: Asking for balances or account limits.

            VALID KNOWLEDGE INTENTS (Requires Policy Lookup):
            - policy_question: General bank rules, limits, timelines.
            - check_fees: Asking what the fees are (not waiving them).
            - general_inquiry: "Hi", "How are you", etc.

            EXAMPLES:
            User: "How much did I spend on Uber last month?" -> Intent: financial_analysis
            User: "I didn't make this $50 Starbucks charge, cancel it!" -> Intent: submit_dispute
            User: "My wallet was stolen!" -> Intent: report_fraud
            User: "What is the overdraft fee?" -> Intent: check_fees
            User: "Can you refund my overdraft fee?" -> Intent: fee_waiver
    """

    ENTITY_SYSTEM = """You are the Entity Extraction engine for Nexus Bank.
            Extract the specific entities mentioned in the customer's message to populate our database tool schemas.

            CRITICAL RULES:
            1. Only extract information EXPLICITLY stated by the user. Do NOT guess or hallucinate. If an entity is not explicitly mentioned, you MUST set it to null.
            2. 'amount' must be a pure float (e.g., 50.0). Strip all currency symbols like '$'.
            3. 'time_period_days' must be an integer. "last month" = 30, "last week" = 7, "yesterday" = 1.
            4. 'expense_category' must be a generic category like 'dining', 'transport', 'retail', 'subscription', 'fee'. DO NOT include trailing commas or spaces.

            EXAMPLES:
            User: "Did I buy anything from BestBuy yesterday?" 
            -> merchant_name: "BestBuy", time_period_days: 1, amount: null, expense_category: null
            
            User: "I want to dispute the $45.99 charge on my card ending in 4321."
            -> amount: 45.99, card_id: "4321", merchant_name: null, expense_category: null
            
            User: "How much did I spend on food last month?"
            -> expense_category: "dining", time_period_days: 30, merchant_name: null, amount: null
            
            User: "What are my last few transactions?"
            -> expense_category: null, time_period_days: null, merchant_name: null, amount: null
    """

    SENTIMENT_SYSTEM = """You are the Sentiment Analysis engine for Nexus Bank.
            Analyze the emotional tone of the customer's message.

            RULES:
            - Score must be a float between -1.0 (extremely angry) and 1.0 (very satisfied).
            - 0.0 is completely neutral.
            - Label MUST be one of: satisfied, neutral, frustrated, angry.

            EXAMPLES:
            User: "Why the hell did you charge me again?!" -> score: -0.9, label: angry
            User: "Can I get a new card?" -> score: 0.0, label: neutral
            User: "Thank you so much, this app is great!" -> score: 0.9, label: satisfied
            User: "I've been waiting for 3 days, this is annoying." -> score: -0.6, label: frustrated
    """

    RISK_SYSTEM = """You are the Risk & Compliance Assessment engine for Nexus Bank.
            Detect critical risk flags and evaluate the complexity of the request.

            FLAGS TO DETECT:
            - 'lawsuit_threat': User mentions lawyers, suing, or legal action against the bank. (DO NOT flag for general complaints).
            - 'fraud_signal': User claims unauthorized access, stolen identity, or unrecognized transactions.
            - 'explicit_human_request': User specifically asks for a "human", "agent", "manager", or "representative".
            - 'suicide_risk' / 'hardship': User threatens self-harm or extreme financial desperation.

            COMPLEXITY (1-3):
            1: Simple factual question or greeting.
            2: Single database action (e.g., freeze card, check balance).
            3: Multi-step investigation or complex math (e.g., fraud investigation, financial analysis).

            EXAMPLES:
            User: "Let me talk to a real person right now!" 
            -> flags: ["explicit_human_request"], complexity: 1
            User: "If you don't refund this, my lawyer will contact you." 
            -> flags: ["lawsuit_threat"], complexity: 1
            User: "Someone hacked my account and spent $500!" 
            -> flags: ["fraud_signal"], complexity: 3
    """

    @observe(as_type="span", name="input_classifier")
    async def run(self, message: str, conversation_history: list = None) -> AgentResult:
        """
        Run all 4 analyses in parallel.
        Total time = slowest single analysis, not sum of all.
        """
        history = conversation_history or []

        # Run all analyses simultaneously
        results = await asyncio.gather(
            self._classify_intent(message, history),
            self._extract_entities(message, history),
            self._analyze_sentiment(message, history),
            self._assess_risk(message, history),
            return_exceptions=True
        )

        intent_result, entity_result, sentiment_result, risk_result = results

        # Handle any exceptions gracefully
        intent_data = self._safe_parse(intent_result, IntentResult(primary_intent="unclear", confidence=0.5))
        entity_data = self._safe_parse(entity_result, EntityResult())
        sentiment_data = self._safe_parse(sentiment_result, SentimentResult(score=0.0, label="neutral"))
        risk_data = self._safe_parse(risk_result, RiskResult(complexity=1, language="en"))

        analyzed = AnalyzedInput(
            raw_message=message,
            primary_intent=intent_data.primary_intent,
            secondary_intent=intent_data.secondary_intent,
            confidence=intent_data.confidence,
            entities=entity_data,
            sentiment=sentiment_data,
            language=risk_data.language,
            complexity=risk_data.complexity,
            risk_flags=risk_data.flags,
            word_count=len(message.split())
        )

        return AgentResult(
            success=True,
            output=analyzed,
            agent_name=self.name,
            model_tier=self.model_tier,
            latency_ms=0, 
            tokens_used=0 
        )

    def _format_history(self, history: list) -> str:
        """Format last N messages into a string for classifier context."""
        if not history:
            return ""
        lines = ["\n--- Recent conversation context ---"]
        for msg in history:
            if isinstance(msg, dict):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
            else:
                # Handle LangChain message objects (HumanMessage, AIMessage)
                role = getattr(msg, "type", "unknown")
                content = getattr(msg, "content", "")
            lines.append(f"{role}: {content}")
        lines.append("--- End of context ---\n")
        return "\n".join(lines)

    async def _classify_intent(self, message: str, history: list) -> dict:
        context = self._format_history(history)
        user_input = f"{context}Current message: {message}" if context else message
        result = await self.call_llm_pydantic(system_prompt=self.INTENT_SYSTEM, user_message=user_input, pydantic_model=IntentResult)
        logger.error(f"[DEBUG INTENT] complete_pydantic returned: {result}")
        return result.get("parsed")

    async def _extract_entities(self, message: str, history: list) -> dict:
        context = self._format_history(history)
        user_input = f"{context}Current message: {message}" if context else message
        result = await self.call_llm_pydantic(system_prompt=self.ENTITY_SYSTEM, user_message=user_input, pydantic_model=EntityResult)
        return result.get("parsed")

    async def _analyze_sentiment(self, message: str, history: list) -> dict:
        context = self._format_history(history)
        user_input = f"{context}Current message: {message}" if context else message
        result = await self.call_llm_pydantic(system_prompt=self.SENTIMENT_SYSTEM, user_message=user_input, pydantic_model=SentimentResult)
        return result.get("parsed")

    async def _assess_risk(self, message: str, history: list) -> dict:
        context = self._format_history(history)
        user_input = f"{context}Current message: {message}" if context else message
        result = await self.call_llm_pydantic(system_prompt=self.RISK_SYSTEM, user_message=user_input, pydantic_model=RiskResult)
        return result.get("parsed")

    def _safe_parse(self, parsed_result, default: Any) -> Any:
        if isinstance(parsed_result, Exception):
            import traceback
            logger.error(f"[SAFE PARSE EXCEPTION] Caught exception during LLM call:")
            traceback.print_exception(type(parsed_result), parsed_result, parsed_result.__traceback__)
            return default
        if parsed_result is None:
            logger.error("[SAFE PARSE] Warning: parsed_result was None. Returning default.")
            return default
        return parsed_result