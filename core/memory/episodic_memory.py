# core/memory/episodic_memory.py

import os
import json
from typing import List, Dict, Any, Optional
from infrastructure.llm.nim_client import nim_client
from langfuse.decorators import observe


class EpisodicMemory:
    """
    Manages user episodic memory via Mem0 API.
    Stores facts learned during conversations and retrieves them
    at the start of future conversations for personalized context.
    
    Falls back gracefully if Mem0 is not configured.
    """

    def __init__(self):
        self.api_key = os.getenv("MEM0_API_KEY")
        self.enabled = self.api_key and self.api_key != "your-mem0-key-here"
        self._client = None

        if self.enabled:
            try:
                from mem0 import MemoryClient
                self._client = MemoryClient(api_key=self.api_key)
            except ImportError:
                print("[EpisodicMemory] mem0 package not installed. Memory disabled.")
                self.enabled = False
            except Exception as e:
                print(f"[EpisodicMemory] Failed to initialize Mem0: {e}")
                self.enabled = False

    @observe(as_type="span", name="memory_retrieve")
    async def retrieve(self, user_id: str, query: str = None) -> List[Dict[str, Any]]:
        """
        Retrieve stored memories for a user.
        Returns a list of memory dicts with 'text' keys.
        """
        if not self.enabled or not self._client:
            return []

        try:
            if query:
                memories = self._client.search(query, user_id=user_id, limit=5)
            else:
                memories = self._client.get_all(user_id=user_id)

            results = []
            for mem in memories:
                if isinstance(mem, dict):
                    results.append({
                        "id": mem.get("id", ""),
                        "text": mem.get("memory", mem.get("text", "")),
                        "created_at": mem.get("created_at", "")
                    })
            return results

        except Exception as e:
            print(f"[EpisodicMemory] Retrieve error: {e}")
            return []

    @observe(as_type="span", name="memory_store")
    async def store(self, user_id: str, conversation_text: str, metadata: dict = None) -> bool:
        """
        Store a memory from this conversation for the user.
        Uses the LLM to extract key facts before storing.
        """
        if not self.enabled or not self._client:
            return False

        try:
            # Extract key facts using the LLM
            extraction = await nim_client.complete(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Extract 1-3 key facts about the customer from this conversation. "
                            "Focus on: products they own, issues they had, preferences, resolution outcomes. "
                            "Return each fact as a short sentence on a new line. "
                            "If there are no important facts, return 'NONE'."
                        )
                    },
                    {"role": "user", "content": conversation_text}
                ],
                tier="fast",
                max_tokens=200
            )

            facts_text = extraction.get("content", "")
            if not facts_text or "NONE" in facts_text.upper():
                return False

            # Store via Mem0
            self._client.add(
                facts_text,
                user_id=user_id,
                metadata=metadata or {}
            )
            return True

        except Exception as e:
            print(f"[EpisodicMemory] Store error: {e}")
            return False

    def format_memories_for_prompt(self, memories: List[Dict[str, Any]]) -> str:
        """Format retrieved memories into a string for LLM context."""
        if not memories:
            return ""

        lines = ["Previous interactions with this customer:"]
        for mem in memories[:5]:
            text = mem.get("text", "")
            if text:
                lines.append(f"  - {text}")

        return "\n".join(lines)


# Singleton
episodic_memory = EpisodicMemory()
