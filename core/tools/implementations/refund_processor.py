# core/tools/implementations/refund_processor.py

import uuid
import random
from datetime import datetime

async def process_refund(order_id: str, amount: float, reason: str) -> dict:
    """
    Mock refund processor (simulates Stripe).
    In production this calls Stripe API.
    """
    
    # Simulate occasional payment gateway failures (2% rate) to test agent error handling
    if random.random() < 0.02:
        return {
            "ok": False,
            "error": "Payment gateway timeout. Please try again later.",
            "data": None
        }

    refund_id = f"REF-{uuid.uuid4().hex[:8].upper()}"

    return {
        "ok": True,
        "error": None,
        "data": {
            "refund_id": refund_id,
            "order_id": order_id,
            "amount": amount,
            "status": "processing",
            "estimated_arrival": "3-5 business days",
            "processed_at": str(datetime.now())
        }
    }