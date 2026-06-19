import asyncio
import json
from core.agents.input_classifier import InputClassifier
from core.agents.action_agent import ActionAgent
from core.quality.judge import QualityJudge

async def debug_queries():
    queries = [
        "How much did I spend on dining last month?",
        "What are my last few transactions?",
        "Did I buy anything from Target recently?",
        "I lost my card ending in 0852, freeze it immediately!",
        "Can you waive the $35 overdraft fee?",
        "I want to dispute the Uber charge.",
        "How much did I spend on food in the last 7 days?",
        "My wallet was stolen!",
        "What is my current account balance?",
        "Can I invest my money in Bitcoin?"
    ]
    
    user_id = "114d5737-7dbf-4013-95bb-5a218ee4071f"
    classifier = InputClassifier("input_classifier", "fast")
    action = ActionAgent()
    judge = QualityJudge()

    for idx, message in enumerate(queries):
        print(f"\n[{idx+1}/10] QUERY: '{message}'")
        
        # 1. Classifier
        res = await classifier.run(message)
        analyzed = res.output
        print(f"  -> INTENT: {analyzed.primary_intent}")
        print(f"  -> ENTITIES: {json.dumps(analyzed.entities.model_dump())}")
        
        # 2. Action Agent
        action_res = await action.run(
            intent=analyzed.primary_intent,
            entities=analyzed.entities.model_dump(),
            user_id=user_id,
            conversation_id=f"test_conv_{idx}",
            user_message=message
        )
        safe_resp = action_res.output["response"].replace('\u202f', ' ').replace('\u2011', '-')
        print(f"  -> AGENT RESPONSE: {safe_resp}")
        
        # 3. Judge
        eval_res = await judge.evaluate(
            user_message=message,
            agent_response=safe_resp,
            tool_results=action_res.output.get("tool_results", []),
            intent=analyzed.primary_intent
        )
        pass_str = "PASS" if eval_res.overall_pass else "FAIL"
        print(f"  -> JUDGE: {pass_str} (Scores: {eval_res.factual_accuracy}, {eval_res.helpfulness}, {eval_res.policy_compliance}, {eval_res.tool_correctness}, {eval_res.conversation_flow})")
        if not eval_res.overall_pass:
            print(f"     REASON: {eval_res.reasoning}")

if __name__ == "__main__":
    asyncio.run(debug_queries())
