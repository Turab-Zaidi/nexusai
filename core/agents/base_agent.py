# core/agents/base_agent.py

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Any

from infrastructure.llm.nim_client import nim_client


@dataclass
class AgentResult:
    """Standard result from any agent"""
    success: bool
    output: Any
    agent_name: str
    model_tier: str
    latency_ms: int
    tokens_used: int
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)


class BaseAgent(ABC):
    """
    Base class for all NexusAI agents.
    Handles NIM calls, timing, error handling.
    Every specialist agent inherits this.
    """

    def __init__(
        self,
        name: str,
        model_tier: str = "standard"
    ):
        self.name = name
        self.model_tier = model_tier
        self.nim = nim_client

    @abstractmethod
    async def run(self, *args, **kwargs) -> AgentResult:
        """Each agent implements its own run method"""
        pass

    async def call_llm(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.1,
        max_tokens: int = 1024
    ) -> dict:
        """Standard LLM call with system + user message"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]

        return await self.nim.complete(
            messages=messages,
            tier=self.model_tier,
            temperature=temperature,
            max_tokens=max_tokens
        )

    async def call_llm_json(
        self,
        system_prompt: str,
        user_message: str
    ) -> dict:
        """LLM call expecting JSON response"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]

        return await self.nim.complete_json(
            messages=messages,
            tier=self.model_tier
        )

    def make_result(
        self,
        success: bool,
        output: Any,
        llm_response: dict,
        error: str = None
    ) -> AgentResult:
        """Create standardized AgentResult"""

        return AgentResult(
            success=success,
            output=output,
            agent_name=self.name,
            model_tier=self.model_tier,
            latency_ms=llm_response.get("latency_ms", 0),
            tokens_used=(
                llm_response.get("prompt_tokens", 0) +
                llm_response.get("completion_tokens", 0)
            ),
            error=error,
            metadata={
                "model": llm_response.get("model"),
            }
        )