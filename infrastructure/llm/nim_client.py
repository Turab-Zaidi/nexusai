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
    Implements key rotation logic across multiple clients.
    """

    def __init__(self):
        keys_str = os.getenv("NVIDIA_API_KEYS", "")
        if keys_str:
            api_keys = [k.strip() for k in keys_str.split(",") if k.strip()]
        else:
            api_keys = [os.getenv("NVIDIA_API_KEY")]

        base_url = os.getenv("NVIDIA_BASE_URL")
        
        self.clients = [
            AsyncOpenAI(api_key=key, base_url=base_url, timeout=15.0)
            for key in api_keys
        ]
        self.current_client_idx = 0

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
        response_format: dict = None,
        tools: list = None,
        tool_choice: dict = None
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
        if tools:
            kwargs["tools"] = tools
        if tool_choice:
            kwargs["tool_choice"] = tool_choice

        try:
            # Select client for key rotation
            client = self.clients[self.current_client_idx]
            self.current_client_idx = (self.current_client_idx + 1) % len(self.clients)

            response = await client.chat.completions.create(**kwargs)
            latency_ms = int((time.time() - start_time) * 1000)

            message_obj = response.choices[0].message
            content = message_obj.content
            tool_calls = message_obj.tool_calls
            
            # Update Langfuse observation
            langfuse_context.update_current_observation(
                model=model,
                input=messages,
                output=content if content else (tool_calls[0].function.arguments if tool_calls else ""),
                usage={
                    "promptTokens": response.usage.prompt_tokens,
                    "completionTokens": response.usage.completion_tokens
                }
            )

            return {
                "content": content,
                "tool_calls": [tc.model_dump() for tc in tool_calls] if tool_calls else None,
                "model": model,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "latency_ms": latency_ms,
                "tier": tier,
                "error": None
            }
        except Exception as e:
            print(f"[NIM CLIENT ERROR] {type(e).__name__}: {str(e)}")
            return {
                "content": None,
                "model": model,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "latency_ms": int((time.time() - start_time) * 1000),
                "tier": tier,
                "error": str(e)
            }

    async def complete_pydantic(
        self,
        messages: list[dict],
        pydantic_model,
        tier: str = "fast",
        temperature: float = 0.0
    ) -> dict:
        """
        Make a completion call using OpenAI Tool Calling (function calling)
        to strictly enforce Pydantic schema output. This is identical to how 
        LangChain's .with_structured_output() works under the hood.
        """
        schema = pydantic_model.model_json_schema()
        
        tools = [{
            "type": "function",
            "function": {
                "name": "structured_output",
                "description": "Output the final result matching this schema",
                "parameters": schema
            }
        }]
        
        tool_choice = {
            "type": "function",
            "function": {"name": "structured_output"}
        }

        result = await self.complete(
            messages=messages,
            tier=tier,
            temperature=temperature,
            max_tokens=512,
            tools=tools,
            tool_choice=tool_choice
        )

        if result["error"]:
            result["parsed"] = None
            return result

        try:
            # Extract arguments from the tool call
            tool_calls = result.get("tool_calls")
            if not tool_calls:
                raise ValueError("Model did not return any tool calls.")
            
            content = tool_calls[0]["function"]["arguments"]
            
            # Validate output via Pydantic
            parsed_model = pydantic_model.model_validate_json(content.strip())
            result["parsed"] = parsed_model
            result["parse_error"] = None
        except Exception as e:
            print(f"[PYDANTIC PARSE ERROR] {str(e)}")
            result["parsed"] = None
            result["parse_error"] = str(e)

        return result


# Singleton instance
nim_client = NIMClient()