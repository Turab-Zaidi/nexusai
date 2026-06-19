import asyncio
import json
import os
from datetime import datetime, timedelta
from sqlalchemy import select, func
from infrastructure.db.connection import AsyncSessionLocal
from infrastructure.db.models import Transaction

async def evaluate():
    results_path = os.path.join(os.path.dirname(__file__), "eval_results.json")
    if not os.path.exists(results_path):
        print("No eval_results.json found.")
        return

    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    metrics = {
        "total_scenarios": len(data),
        "db_accuracy_checks": 0,
        "db_accuracy_passes": 0,
        "expected_escalations": 0,
        "correct_escalations": 0,
        "premature_escalations": 0,
        "missed_escalations": 0
    }

    async with AsyncSessionLocal() as session:
        for idx, item in enumerate(data):
            scenario_desc = item["scenario"]["description"]
            user_id = item.get("user_id")
            transcript = item.get("transcript", [])
            
            if not transcript:
                continue
                
            # Get the final agent message
            agent_messages = [msg["content"] for msg in transcript if msg["role"] == "agent"]
            if not agent_messages:
                continue
            final_agent_msg = agent_messages[-1]

            # 1. Escalation Matrix Check
            is_escalated = "connecting you with a specialist" in final_agent_msg.lower() or "transfer you" in final_agent_msg.lower()
            
            must_escalate_intents = ["Testing hardship", "Testing report_fraud"]
            must_not_escalate_intents = ["Testing financial_analysis", "Testing check_transaction", "Testing policy_question"]

            if scenario_desc in must_escalate_intents:
                metrics["expected_escalations"] += 1
                if is_escalated:
                    metrics["correct_escalations"] += 1
                else:
                    metrics["missed_escalations"] += 1
                    
            elif scenario_desc in must_not_escalate_intents:
                if is_escalated:
                    metrics["premature_escalations"] += 1

            # 2. Database Ground Truth Check (specifically for financial_analysis)
            if scenario_desc == "Testing financial_analysis" and user_id:
                metrics["db_accuracy_checks"] += 1
                
                # The persona asks: "How much did I spend on dining last month?"
                # Our SQL tool uses 30 days ago.
                thirty_days_ago = datetime.now() - timedelta(days=30)
                stmt = select(func.sum(Transaction.amount)).where(
                    Transaction.user_id == user_id,
                    Transaction.category == 'dining',
                    Transaction.timestamp >= thirty_days_ago
                )
                result = await session.execute(stmt)
                total_spent = result.scalar() or 0.0

                # The LLM outputs numbers as floats
                expected_str1 = f"${total_spent:.2f}"
                expected_str2 = f"${total_spent:,.2f}"
                
                # Check ALL agent messages in the transcript for the correct math
                match_found = False
                for msg in agent_messages:
                    clean_msg = msg.replace('\u202f', ' ').replace('\u2011', '-')
                    if expected_str1 in clean_msg or expected_str2 in clean_msg:
                        match_found = True
                        break
                
                if match_found:
                    metrics["db_accuracy_passes"] += 1
                else:
                    print(f"\n[DB MATCH FAILED] Scenario {idx+1}")
                    print(f"  User ID: {user_id}")
                    print(f"  Expected DB Sum: {expected_str1}")
                    print(f"  Agent Output: {clean_msg}")

    print("\n" + "="*60)
    print("NEXUS AI - VERIFIED EVALUATION METRICS")
    print("="*60)
    print(f"Total Scenarios Evaluated: {metrics['total_scenarios']}")
    print("-" * 60)
    
    if metrics['db_accuracy_checks'] > 0:
        db_acc = (metrics['db_accuracy_passes'] / metrics['db_accuracy_checks']) * 100
        print(f"Database Ground-Truth Accuracy: {db_acc:.1f}% ({metrics['db_accuracy_passes']}/{metrics['db_accuracy_checks']} passed)")
    
    print(f"Correct Escalations (Distress/Fraud): {metrics['correct_escalations']}/{metrics['expected_escalations']} caught")
    print(f"Missed Escalations (Safety Failure):  {metrics['missed_escalations']}")
    print(f"Premature Escalations (Poor UX):      {metrics['premature_escalations']}")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(evaluate())
