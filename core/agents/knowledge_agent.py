# core/agents/knowledge_agent.py
"""
KnowledgeAgent — queries the FAISS vector store built from the
Nexus Bank Operations & Policy Manual (40-page PDF).
Uses HuggingFace embeddings (all-MiniLM-L6-v2) to match the
embedding model used during ingestion on Kaggle.
"""

import os
import json
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from .base_agent import BaseAgent, AgentResult


FAISS_INDEX_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "faiss_db"
)


class KnowledgeAgent(BaseAgent):
    """
    Answers policy/fee/regulation questions by searching the
    FAISS index built from the Nexus Bank Policy Manual.
    Injects user_context (past tickets) for personalised answers.
    """

    def __init__(self):
        super().__init__(name="knowledge_agent", model_tier="standard")
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore = FAISS.load_local(
            FAISS_INDEX_PATH,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}   # Retrieve top-4 policy chunks
        )

    async def run(
        self,
        user_message: str,
        intent: str,
        user_context: str = ""
    ) -> AgentResult:


        # ── Retrieve relevant policy chunks from FAISS ────────────────────
        docs = self.retriever.invoke(user_message)

        if not docs:
            return AgentResult(
                success=True,
                output={
                    "response": (
                        "I wasn't able to find specific policy information on that topic. "
                        "Please contact a Nexus Bank specialist for further assistance."
                    ),
                    "sources": [],
                    "grounded": False
                },
                agent_name=self.name,
                model_tier=self.model_tier,
                latency_ms=0,
                tokens_used=0
            )

        # ── Build context from retrieved chunks ───────────────────────────
        policy_context = "\n\n---\n\n".join([doc.page_content for doc in docs])
        sources = [
            doc.metadata.get("Header 2") or doc.metadata.get("Header 1", "Policy Manual")
            for doc in docs
        ]

        history_block = f"\nCUSTOMER HISTORY:\n{user_context}\n" if user_context else ""

        system_prompt = f"""You are a professional Nexus Bank support agent with deep knowledge of bank policy.
Your job is to answer the customer's question using ONLY the POLICY CONTEXT provided below.

CRITICAL COMPLIANCE RULES:
1. Anti-Hallucination: If the answer is NOT explicitly stated in the POLICY CONTEXT, you must say: "I do not have access to that information. Please contact a specialist." Do not guess.
2. No Promises: Never guarantee specific outcomes (e.g., "you will definitely get a refund").
3. Citation: Always cite the relevant policy section name if it is provided in the text.
4. Tone: Professional, empathetic, under 120 words.

EXAMPLES:
Context: "Overdraft fees are $35 per occurrence."
User: "How much is the overdraft fee?"
Response: "According to our policy, the overdraft fee is $35 per occurrence."

Context: "Overdraft fees are $35."
User: "Can you waive my fee?"
Response: "I cannot guarantee a fee waiver, but I can connect you with a specialist to review your account."

Context: "Overdraft fees are $35."
User: "What are your wire transfer limits?"
Response: "I do not have access to wire transfer information in my current policies. Please contact a specialist for assistance."
{history_block}
POLICY CONTEXT:
{policy_context}"""

        response = await self.call_llm(
            system_prompt=system_prompt,
            user_message=user_message,
            temperature=0.1
        )

        if response.get("error"):
            return self.make_result(
                success=False,
                output={
                    "response": f"System Error: Failed to contact AI provider. ({response['error']})",
                    "sources": [],
                    "grounded": False
                },
                llm_response=response
            )

        return self.make_result(
            success=True,
            output={
                "response": response.get("content") or "Error: Blank response from AI",
                "sources": sources,
                "grounded": True
            },
            llm_response=response
        )