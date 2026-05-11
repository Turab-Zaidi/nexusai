# core/agents/input_classifier.py

import asyncio
from dataclasses import dataclass, field
from typing import Optional

from .base_agent import BaseAgent, AgentResult


@dataclass
class SentimentResult:
    score: float      # -1.0 to 1.0
    label: str        # neutral, frustrated, angry, satisfied


@dataclass
class EntityResult:
    order_id: Optional[str] = None
    product_name: Optional[str] = None
    amount: Optional[float] = None
    date_mentioned: Optional[str] = None
    customer_name: Optional[str] = None


@dataclass
class AnalyzedInput:
    raw_message: str
    primary_intent: str
    secondary_intent: Optional[str]
    confidence: float
    entities: EntityResult
    sentiment: SentimentResult
    language: str
    complexity: int       
    risk_flags: list[str]
    word_count: int


class InputClassifier(BaseAgent):
    """
    Analyzes raw customer messages.
    Runs 4 analyses in parallel using asyncio.gather.
    Uses fast tier (Llama 3.1 8B) for speed.
    """

    INTENT_SYSTEM = """You classify customer support messages.
            Return JSON with these exact fields:
            {
            "primary_intent": "one of the valid intents",
            "secondary_intent": "another intent or null",
            "confidence": 0.0 to 1.0
            }

            Valid intents: cancel_order, change_order, check_invoice,
            check_refund_policy, complaint, contact_human_agent,
            delivery_period, get_invoice, get_refund, payment_issue,
            place_order, track_order, track_refund, unclear

            Pick the single best matching intent.
    """

    ENTITY_SYSTEM = """Extract entities from customer messages.
            Return JSON exactly like this:
            {
            "order_id": "order number or null",
            "product_name": "product name or null",
            "amount": numeric amount or null,
            "date_mentioned": "date string or null",
            "customer_name": "name or null"
            }
            Only extract what is explicitly mentioned. If not found, use null.
    """

    SENTIMENT_SYSTEM = """Analyze the sentiment of this message.
            Return JSON exactly like this:
            {
            "score": -1.0 to 1.0,
            "label": "one of: satisfied, neutral, frustrated, angry"
            }
            -1.0 = extremely angry, 0.0 = neutral, 1.0 = very satisfied
    """

    RISK_SYSTEM = """Check for risk flags in this message.
            Return JSON exactly like this:
            {
            "flags": [],
            "complexity": 1,
            "language": "en"
            }
            Flags can include: contains_pii, legal_language, fraud_signal, high_value_transaction, explicit_human_request
            Complexity: 1=simple lookup, 2=needs tool call, 3=multi-step
            Language: ISO 639-1 code (en, es, fr, ar, etc)
    """

    async def run(self, message: str) -> AgentResult:
        """
        Run all 4 analyses in parallel.
        Total time = slowest single analysis, not sum of all.
        """

        # Run all analyses simultaneously
        results = await asyncio.gather(
            self._classify_intent(message),
            self._extract_entities(message),
            self._analyze_sentiment(message),
            self._assess_risk(message),
            return_exceptions=True
        )

        intent_result, entity_result, sentiment_result, risk_result = results

        # Handle any exceptions gracefully
        intent_data = self._safe_parse(intent_result, {
            "primary_intent": "unclear", "secondary_intent": None, "confidence": 0.5
        })

        entity_data = self._safe_parse(entity_result, {})

        sentiment_data = self._safe_parse(sentiment_result, {
            "score": 0.0, "label": "neutral"
        })

        risk_data = self._safe_parse(risk_result, {
            "flags": [], "complexity": 1, "language": "en"
        })

        analyzed = AnalyzedInput(
            raw_message=message,
            primary_intent=intent_data.get("primary_intent", "unclear"),
            secondary_intent=intent_data.get("secondary_intent"),
            confidence=intent_data.get("confidence", 0.5),
            entities=EntityResult(
                order_id=entity_data.get("order_id"),
                product_name=entity_data.get("product_name"),
                amount=entity_data.get("amount"),
                date_mentioned=entity_data.get("date_mentioned"),
                customer_name=entity_data.get("customer_name")
            ),
            sentiment=SentimentResult(
                score=sentiment_data.get("score", 0.0),
                label=sentiment_data.get("label", "neutral")
            ),
            language=risk_data.get("language", "en"),
            complexity=risk_data.get("complexity", 1),
            risk_flags=risk_data.get("flags", []),
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

    async def _classify_intent(self, message: str) -> dict:
        result = await self.call_llm_json(system_prompt=self.INTENT_SYSTEM, user_message=message)
        return result.get("parsed", {}) or {}

    async def _extract_entities(self, message: str) -> dict:
        result = await self.call_llm_json(system_prompt=self.ENTITY_SYSTEM, user_message=message)
        return result.get("parsed", {}) or {}

    async def _analyze_sentiment(self, message: str) -> dict:
        result = await self.call_llm_json(system_prompt=self.SENTIMENT_SYSTEM, user_message=message)
        return result.get("parsed", {}) or {}

    async def _assess_risk(self, message: str) -> dict:
        result = await self.call_llm_json(system_prompt=self.RISK_SYSTEM, user_message=message)
        return result.get("parsed", {}) or {}

    def _safe_parse(self, result, default: dict) -> dict:
        if isinstance(result, Exception):
            return default
        if isinstance(result, dict):
            return result
        return default