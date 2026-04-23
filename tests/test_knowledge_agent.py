# tests/test_knowledge_agent.py

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from core.agents.knowledge_agent import KnowledgeAgent

async def test_agent():
    print("Loading Knowledge Agent...")
    agent = KnowledgeAgent()
    
    # Test 1: Simulate correct intent for a refund policy question
    question_1 = "What is your refund policy? How many days do I have?"
    intent_1 = "check_refund_policy"
    print(f"\nUser: '{question_1}' (Intent: {intent_1})")
    
    result_1 = await agent.run(user_message=question_1, intent=intent_1)
    
    print(f" Agent: {result_1.output['response']}")
    print(f" Sources Used: {result_1.output['sources']}")
    
    # Test 2: Simulate an unclear intent for an out-of-scope question
    question_2 = "What is the capital of France?"
    intent_2 = "unclear"
    print(f"\nUser: '{question_2}' (Intent: {intent_2})")
    
    result_2 = await agent.run(user_message=question_2, intent=intent_2)
    
    print(f" Agent: {result_2.output['response']}")
    print(f" Sources Used: {result_2.output['sources']}")

if __name__ == "__main__":
    asyncio.run(test_agent())