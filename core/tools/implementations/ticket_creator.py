# core/tools/implementations/ticket_creator.py

import random
from datetime import datetime
from langfuse.decorators import observe

@observe(as_type="span", name="create_ticket")
async def create_ticket(user_id: str, conversation_id: str, issue_summary: str, priority: str = "normal") -> dict:
    """
    Mock ticket creator (simulates Zendesk).
    """
    ticket_id = f"TKT-{random.randint(10000, 99999)}"

    return {
        "ok": True,
        "error": None,
        "data": {
            "ticket_id": ticket_id,
            "status": "open",
            "priority": priority,
            "summary": issue_summary,
            "estimated_response": "1-2 hours" if priority == "high" else "4-8 hours",
            "created_at": str(datetime.now())
        }
    }