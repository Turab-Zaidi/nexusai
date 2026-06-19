import sys
import builtins

# Mute console crashes from state_machine.py's internal logs
_original_print = builtins.print
def safe_print(*args, **kwargs):
    safe_args = [arg.encode('ascii', 'replace').decode('ascii') if isinstance(arg, str) else arg for arg in args]
    _original_print(*safe_args, **kwargs)
builtins.print = safe_print

import asyncio
import uuid
import json
import os
import random
from infrastructure.llm.nim_client import nim_client
from core.orchestrator.state_machine import create_graph

# Add DB imports to fetch real users
from infrastructure.db.connection import AsyncSessionLocal
from infrastructure.db.models import User
from sqlalchemy import select

with open(os.path.join(os.path.dirname(__file__), "eval_personas.json"), "r") as f:
    SCENARIOS = json.load(f)

async def get_real_user_ids():
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(User.id))
        user_ids = [row[0] for row in result.all()]
        return user_ids

async def simulate_conversation(graph, scenario: dict, real_user_id: str, max_turns: int = 3):
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    state = {
        "user_id": real_user_id,
        "conversation_id": thread_id,
        "messages": [],
        "turn_count": 0
    }
    
    chat_history = []
    
    for turn in range(max_turns):
        prompt = f"""{scenario['persona']}
        
Current Chat History:
{json.dumps(chat_history, indent=2)}

Generate your next message to the bank agent. ONLY output the message text, nothing else."""

        sim_response = await nim_client.complete(
            messages=[{"role": "user", "content": prompt}],
            tier="fast"
        )
        user_message = sim_response.get("content", "").strip()
        chat_history.append({"role": "user", "content": user_message})
        
        # Inject the base state on the very first turn, otherwise just pass the new message
        payload = state if turn == 0 else {}
        payload["current_message"] = user_message
        
        current_state = await graph.ainvoke(payload, config)
        
        agent_response = current_state.get("agent_response", "")
        chat_history.append({"role": "agent", "content": agent_response})
        
        if current_state.get("escalated"):
            break
            
        await asyncio.sleep(2.0)
        
    return chat_history

async def main():
    print("Fetching real user IDs from PostgreSQL...")
    user_ids = await get_real_user_ids()
    if not user_ids:
        print("ERROR: Database is empty. Please run seed_db.py first.")
        return
        
    graph = create_graph()
    results = []
    
    total = len(SCENARIOS)
    print(f"Starting batch evaluation of {total} scenarios...")
    
    for idx, scenario in enumerate(SCENARIOS):
        print(f"[{idx+1}/{total}] Running Scenario: {scenario['description']}")
        
        try:
            # Pick a completely random real user for this scenario
            random_user_id = random.choice(user_ids)
            transcript = await simulate_conversation(graph, scenario, random_user_id)
            results.append({
                "scenario": scenario,
                "user_id": random_user_id,
                "transcript": transcript
            })
        except Exception as e:
            print(f"Error on scenario {idx+1}: {e}")
            
    out_path = os.path.join(os.path.dirname(__file__), "eval_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\nAll simulations complete! Full transcripts saved to: {out_path}")

if __name__ == "__main__":
    asyncio.run(main())
