# tests/test_nim.py

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from infrastructure.llm.nim_client import nim_client

async def test_nim_connection():
    print("Testing NVIDIA NIM connection...")

    result = await nim_client.complete(
        messages=[
            {
                "role": "user",
                "content": "Say hello in exactly 5 words."
            }
        ],
        tier="fast"
    )

    if result.get("error"):
        print(f"FAILED: {result['error']}")
        return

    print(f"Response: {result['content']}")
    print(f"Model: {result['model']}")
    print(f"Tokens: {result['prompt_tokens']} prompt, {result['completion_tokens']} completion")
    print(f"Latency: {result['latency_ms']}ms")
    print("NIM connection: SUCCESS\n")


async def test_json_response():
    print("Testing JSON response...")

    result = await nim_client.complete_json(
        messages=[
            {
                "role": "system",
                "content": "You classify customer support intents."
            },
            {
                "role": "user",
                "content": (
                    'Classify this message: '
                    '"I want a refund for order 12345"\n'
                    'Return: {"intent": "...", "confidence": 0.0}'
                )
            }
        ],
        tier="fast"
    )

    if result.get("parse_error"):
        print(f"JSON Parse Failed: {result['parse_error']}")
        print(f"Raw Content: {result['content']}")
        return

    print(f"Parsed JSON: {result['parsed']}")
    print("JSON response: SUCCESS")


if __name__ == "__main__":
    asyncio.run(test_nim_connection())
    asyncio.run(test_json_response())