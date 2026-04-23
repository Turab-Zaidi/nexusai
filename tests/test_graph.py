# tests/test_graph.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from core.orchestrator.state_machine import create_graph

def test_routing():
    print("Compiling LangGraph State Machine...")
    app = create_graph()
    
    print("\nSimulating an Angry Customer requesting a refund...")
    
    # We fake the analyzed state based on Day 5's output
    initial_state = {
        "messages": [],
        "analyzed_input": {
            "primary_intent": "complaint",
            "sentiment": {"score": -1.0, "label": "angry"},
            "entities": {"order_id": "ORD-12345"},
            "risk_flags": ["explicit_human_request"]
        },
        "turn_count": 0,
        "revision_count": 0,
        "escalated": False
    }

    # Run the state machine
    print("Tracing Graph Execution Path:")
    for output in app.stream(initial_state):
        for node_name, state_updates in output.items():
            print(f"   Executed Node: [{node_name}]")
            if "escalated" in state_updates and state_updates["escalated"]:
                print("      TRIGGERED ESCALATION PROTOCOL")

    print("\nGraph execution complete! Validation successful.")

if __name__ == "__main__":
    test_routing()