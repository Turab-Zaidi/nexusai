# apps/api/routes/chat.py

import uuid
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from core.agents.input_classifier import InputClassifier

router = APIRouter()

# Instantiate the classifier once when the router loads
classifier = InputClassifier("input_classifier", "fast")

class ChatRequest(BaseModel):
    message: str
    session_id: str = None
    user_id: str = None

@router.post("/chat")
async def chat(request: ChatRequest):
    """
    Main chat endpoint.
    Week 1 version: Returns the AI analyzed input.
    Will be expanded to the full state machine in Week 2.
    """

    session_id = request.session_id or str(uuid.uuid4())
    user_id = request.user_id or "anonymous"

    print(f"\n[API] Received message from {user_id}: {request.message}")

    # Run input analysis (The 4 parallel LLM calls)
    analysis_result = await classifier.run(request.message)
    
    if not analysis_result.success:
        raise HTTPException(status_code=500, detail="Failed to analyze input")

    analyzed = analysis_result.output

    return {
        "session_id": session_id,
        "message_received": request.message,
        "analysis": {
            "intent": analyzed.primary_intent,
            "confidence": analyzed.confidence,
            "sentiment": analyzed.sentiment.label,
            "sentiment_score": analyzed.sentiment.score,
            "entities": {
                "order_id": analyzed.entities.order_id,
                "product": analyzed.entities.product_name,
                "amount": analyzed.entities.amount
            },
            "complexity": analyzed.complexity,
            "risk_flags": analyzed.risk_flags,
            "language": analyzed.language
        },
        "status": "analysis_complete"
    }