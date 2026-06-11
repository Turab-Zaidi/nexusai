# infrastructure/llm/nim_client.py

import os
import time
import json
from openai import AsyncOpenAI
from dotenv import load_dotenv
from langfuse.decorators import observe, langfuse_context

load_dotenv()


class NIMClient:
    """
    Wrapper around NVIDIA NIM API.
    NIM is OpenAI compatible so we use the
    OpenAI client with a different base URL.
    All agents use this single client.
    """

    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=os.getenv("NVIDIA_API_KEY"),
            base_url=os.getenv("NVIDIA_BASE_URL")
        )

        self.models = {
            "fast": os.getenv("FAST_MODEL"),
            "standard": os.getenv("STANDARD_MODEL"),
            "heavy": os.getenv("HEAVY_MODEL")
        }

    @observe(as_type="generation")
    async def complete(
        self,
        messages: list[dict],
        tier: str = "standard",
        temperature: float = 0.1,
        max_tokens: int = 1024,
        response_format: dict = None
    ) -> dict:
        """
        Make a completion call.
        Returns dict with content, usage, latency.
        """

        model = self.models.get(tier, self.models["standard"])
        start_time = time.time()

        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if response_format:
            kwargs["response_format"] = response_format

        try:
            response = await self.client.chat.completions.create(**kwargs)
            latency_ms = int((time.time() - start_time) * 1000)

            # Update Langfuse observation
            langfuse_context.update_current_observation(
                model=model,
                input=messages,
                output=response.choices[0].message.content,
                usage={
                    "promptTokens": response.usage.prompt_tokens,
                    "completionTokens": response.usage.completion_tokens
                }
            )

            return {
                "content": response.choices[0].message.content,
                "model": model,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "latency_ms": latency_ms,
                "tier": tier,
                "error": None
            }
        except Exception as e:
            return {
                "content": None,
                "model": model,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "latency_ms": int((time.time() - start_time) * 1000),
                "tier": tier,
                "error": str(e)
            }

    async def complete_json(
        self,
        messages: list[dict],
        tier: str = "fast",
        temperature: float = 0.0
    ) -> dict:
        """
        Make a completion call expecting JSON output.
        Used for classification and structured extraction.
        """

        # Add JSON instruction to last user message
        messages = messages.copy()
        messages[-1]["content"] = (
            messages[-1]["content"] +
            "\n\nRespond with valid JSON only. No markdown."
        )

        result = await self.complete(
            messages=messages,
            tier=tier,
            temperature=temperature,
            max_tokens=512
        )

        if result["error"]:
            result["parsed"] = None
            return result

        try:
            content = result["content"].strip()
            # Remove markdown code blocks if present (e.g. ```json ... ```)
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            
            result["parsed"] = json.loads(content.strip())
            result["parse_error"] = None
        except json.JSONDecodeError as e:
            result["parsed"] = None
            result["parse_error"] = str(e)

        return result


# Singleton instance
nim_client = NIMClient()