# core/agents/knowledge_agent.py

import json
from sentence_transformers import SentenceTransformer
from sqlalchemy import select
from infrastructure.db.connection import AsyncSessionLocal
from infrastructure.db.models import KnowledgeEntry
from .base_agent import BaseAgent, AgentResult
from langfuse.decorators import observe

class KnowledgeAgent(BaseAgent):
    """
    Answers informational questions using Intent-Filtered RAG.
    It PRE-FILTERS the knowledge base by the detected intent,
    then performs a vector search on that small subset.
    This is much more accurate and faster.
    """

    def __init__(self):
        super().__init__(
            name="knowledge_agent",
            model_tier="standard"
        )
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

    @observe(as_type="span", name="knowledge_agent")
    async def run(
        self,
        user_message: str,
        intent: str, # We now REQUIRE the intent from the classifier
        conversation_history: list = None
    ) -> AgentResult:

        # Step 1: Check if the intent is a knowledge-based one.
        # If not, or if it's "unclear", we don't even search.
        knowledge_intents = [
            "check_refund_policy", "delivery_options",
            "delivery_period", "check_payment_methods",
            "check_cancellation_fee", "review", "complaint"
        ]
        
        if intent not in knowledge_intents or intent == "unclear":
            # DETERMINISTIC GUARDRAIL: Return a hardcoded response for out-of-scope questions.
            # We don't even call the LLM, making it 100% safe.
            return AgentResult(
                success=True,
                output={
                    "response": "I apologize, but I can only assist with customer support questions about orders, deliveries, and our policies. How can I help you with that?",
                    "sources": [],
                    "grounded": False
                },
                agent_name=self.name,
                model_tier=self.model_tier,
                latency_ms=0,
                tokens_used=0
            )

        # Step 2: Get relevant knowledge, FILTERED BY INTENT.
        chunks = await self._retrieve(user_message, intent, top_k=1)

        # If no chunks are found even with the right intent, apologize.
        if not chunks:
            return AgentResult(
                success=True,
                output={
                    "response": "I apologize, but I was unable to find specific information on that topic in our knowledge base.",
                    "sources": [],
                    "grounded": False
                },
                agent_name=self.name,
                model_tier=self.model_tier,
                latency_ms=0,
                tokens_used=0
            )

        # Step 3: Build context from retrieved chunks
        context = f"COMPANY POLICY on {intent}:\n{chunks[0]['answer']}"

        # Step 4: Generate grounded response using the retrieved context
        system_prompt = f"""You are a helpful, professional customer support agent.
Answer the customer's question using ONLY the information provided in the COMPANY POLICY below.

If the answer is not in the context, say so clearly. Do not invent information.
Keep your response concise, polite, and directly address the user's question.

COMPANY POLICY:
{context}"""

        response = await self.call_llm(
            system_prompt=system_prompt,
            user_message=user_message,
            temperature=0.1
        )

        return self.make_result(
            success=True,
            output={
                "response": response["content"],
                "sources": [c["intent"] for c in chunks],
                "grounded": True
            },
            llm_response=response
        )

    async def _retrieve(
        self,
        query: str,
        intent: str, # We now require intent for filtering
        top_k: int = 1
    ) -> list:
        """
        Search pgvector for similar entries, but only within the specified intent.
        """
        query_embedding = self.embedder.encode(query).tolist()

        async with AsyncSessionLocal() as session:
            results = await session.execute(
                select(
                    KnowledgeEntry,
                    KnowledgeEntry.embedding.cosine_distance(query_embedding).label("distance")
                )
                .filter(KnowledgeEntry.intent == intent) # <-- THE MAGIC HAPPENS HERE
                .order_by("distance")
                .limit(top_k)
            )
            entries = results.all()

        if not entries:
            return []

        return [
            {
                "question": entry.KnowledgeEntry.question,
                "answer": entry.KnowledgeEntry.answer,
                "category": entry.KnowledgeEntry.category,
                "intent": entry.KnowledgeEntry.intent,
                "score": 1 - entry.distance
            }
            for entry in entries
        ]