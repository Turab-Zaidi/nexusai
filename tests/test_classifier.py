# tests/test_classifier.py

import asyncio
import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from core.agents.input_classifier import InputClassifier

async def test_classification():
    print("Initializing Input Classifier...")
    # We use the 'fast' model (Llama 8B) because we need pure speed for routing
    clf = InputClassifier("input_classifier", "fast")
    
    test_message = "I am absolutely furious! Where is my refund for order ORD-12345? I will sue you if I don't get my $99.99 back right now!"
    
    print(f"\nAnalyzing message: '{test_message}'")
    print("Running 4 LLM calls in parallel...\n")
    
    start_time = time.time()
    
    result = await clf.run(test_message)
    
    end_time = time.time()
    ai = result.output
    
    print(f"  Total Time: {end_time - start_time:.2f} seconds")
    print("-" * 40)
    print(f" Intent:      {ai.primary_intent} (Confidence: {ai.confidence})")
    print(f" Entities:    Order: {ai.entities.order_id} | Amount: ${ai.entities.amount}")
    print(f" Sentiment:   {ai.sentiment.label.upper()} (Score: {ai.sentiment.score})")
    print(f"  Risk Flags:  {ai.risk_flags}")
    print(f" Complexity:  Level {ai.complexity}")
    print("-" * 40)

if __name__ == "__main__":
    asyncio.run(test_classification())