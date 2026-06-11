# core/guardrails/guardrails.py

import re
from typing import Tuple
from langfuse.decorators import observe


class Guardrails:
    """
    Safety layer that runs BEFORE and AFTER the pipeline.
    - Input guardrails: PII detection, injection detection, scope enforcement
    - Output guardrails: PII redaction in responses
    """

    # ── PII Patterns ──────────────────────────────────────────
    PII_PATTERNS = {
        "credit_card": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "phone": r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
    }

    # ── Injection Patterns ────────────────────────────────────
    INJECTION_PATTERNS = [
        r"ignore (?:all )?(?:previous |above |prior )?instructions",
        r"you are now (?:a |an )",
        r"pretend (?:you are|to be)",
        r"disregard (?:your |all )?(?:rules|guidelines|instructions)",
        r"system prompt",
        r"reveal your (?:instructions|prompt|rules)",
        r"act as (?:a |an )?(?:different|new)",
        r"override (?:your |all )?(?:settings|rules)",
    ]

    # ── Out-of-scope Topics ───────────────────────────────────
    OUT_OF_SCOPE_PATTERNS = [
        r"\b(?:medical|health|diagnosis|prescription|symptom)\b.*\b(?:advice|help|recommend)\b",
        r"\b(?:legal|lawyer|sue|lawsuit|court)\b.*\b(?:advice|help|recommend)\b",
        r"\b(?:invest|stock|crypto|trading)\b.*\b(?:advice|help|recommend)\b",
        r"\bwrite (?:me )?(?:a |an )?(?:essay|poem|story|code|script)\b",
    ]

    @observe(as_type="span", name="guardrails_input")
    async def check_input(self, message: str) -> Tuple[bool, str, dict]:
        """
        Run all input guardrails.
        Returns: (allowed: bool, reason: str, metadata: dict)
        """
        metadata = {
            "pii_detected": False,
            "injection_detected": False,
            "out_of_scope": False,
            "pii_types": []
        }

        # 1. Check for prompt injection attempts
        message_lower = message.lower()
        for pattern in self.INJECTION_PATTERNS:
            if re.search(pattern, message_lower):
                metadata["injection_detected"] = True
                return (
                    False,
                    "I'm sorry, but I can only help with customer support questions. "
                    "Could you please rephrase your request?",
                    metadata
                )

        # 2. Check for out-of-scope requests
        for pattern in self.OUT_OF_SCOPE_PATTERNS:
            if re.search(pattern, message_lower):
                metadata["out_of_scope"] = True
                return (
                    False,
                    "I appreciate your question, but I'm specifically designed for customer support. "
                    "I can help with orders, refunds, shipping, and account questions. "
                    "For other topics, please consult the appropriate professional.",
                    metadata
                )

        # 3. Detect PII in input (flag but don't block — the user may need to share order info)
        for pii_type, pattern in self.PII_PATTERNS.items():
            if re.search(pattern, message):
                metadata["pii_detected"] = True
                metadata["pii_types"].append(pii_type)

        return (True, "", metadata)

    @observe(as_type="span", name="guardrails_output")
    async def sanitize_output(self, response: str) -> str:
        """
        Redact any PII that might appear in the AI's response.
        The AI should never echo back credit cards, SSNs, etc.
        """
        sanitized = response

        # Redact credit card numbers
        sanitized = re.sub(
            r"\b(\d{4})[-\s]?\d{4}[-\s]?\d{4}[-\s]?(\d{4})\b",
            r"\1-****-****-\2",
            sanitized
        )

        # Redact SSNs
        sanitized = re.sub(
            r"\b\d{3}-\d{2}-(\d{4})\b",
            r"***-**-\1",
            sanitized
        )

        return sanitized


# Singleton
guardrails = Guardrails()
