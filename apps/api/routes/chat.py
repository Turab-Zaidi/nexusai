# apps/api/routes/chat.py

import uuid
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from core.orchestrator.pipeline import pipeline

router = APIRouter()


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

    # Run the full pipeline
    try:
        response_data = await pipeline.process(
            message=request.message,
            session_id=session_id,
            user_id=user_id
        )
        return response_data
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))