# apps/api/routes/chat.py

import uuid
import json
import asyncio
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from core.orchestrator.pipeline import pipeline
from core.guardrails.guardrails import guardrails

router = APIRouter()


class ChatRequest(BaseModel):
    message: str
    session_id: str = None
    user_id: str = None


@router.post("/chat")
async def chat(request: ChatRequest):
    """
    Main chat endpoint. Returns full JSON response.
    Includes guardrails check before pipeline execution.
    """
    session_id = request.session_id or str(uuid.uuid4())
    user_id = request.user_id or "anonymous"

    print(f"\n[API] Received message from {user_id}: {request.message}")

    # Run input guardrails
    allowed, rejection_reason, guard_meta = await guardrails.check_input(request.message)
    if not allowed:
        return {
            "session_id": session_id,
            "message_received": request.message,
            "response": rejection_reason,
            "escalated": False,
            "escalation_reason": None,
            "handoff_package": None,
            "agent_used": "guardrails",
            "tools_called": [],
            "analysis": {},
            "guardrails": guard_meta,
            "status": "blocked"
        }

    # Run the full pipeline
    try:
        response_data = await pipeline.process(
            message=request.message,
            session_id=session_id,
            user_id=user_id
        )

        # Sanitize output
        if response_data.get("response"):
            response_data["response"] = await guardrails.sanitize_output(
                response_data["response"]
            )

        response_data["guardrails"] = guard_meta
        return response_data
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    SSE streaming endpoint.
    Streams status updates and the final response token-by-token.
    """
    session_id = request.session_id or str(uuid.uuid4())
    user_id = request.user_id or "anonymous"

    # Check guardrails first
    allowed, rejection_reason, guard_meta = await guardrails.check_input(request.message)

    async def event_generator():
        if not allowed:
            yield f"data: {json.dumps({'type': 'blocked', 'content': rejection_reason})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return

        # Stream status updates as the pipeline runs
        yield f"data: {json.dumps({'type': 'status', 'content': 'Analyzing your message...'})}\n\n"
        await asyncio.sleep(0.1)

        try:
            response_data = await pipeline.process(
                message=request.message,
                session_id=session_id,
                user_id=user_id
            )

            agent_used = response_data.get("agent_used", "unknown")
            status_messages = {
                "knowledge_agent": "Searching our knowledge base...",
                "action_agent": "Processing your request...",
                "resolution_agent": "Working on a multi-step resolution...",
                "escalation_agent": "Preparing handoff to support team...",
            }
            status = status_messages.get(agent_used, "Processing...")
            yield f"data: {json.dumps({'type': 'status', 'content': status})}\n\n"

            # Sanitize output
            full_response = response_data.get("response", "")
            if full_response:
                full_response = await guardrails.sanitize_output(full_response)

            # Stream the response word by word
            words = full_response.split(" ")
            for i, word in enumerate(words):
                token = word if i == len(words) - 1 else word + " "
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
                await asyncio.sleep(0.03)  # Simulate streaming delay

            # Send metadata
            yield f"data: {json.dumps({'type': 'metadata', 'agent_used': agent_used, 'escalated': response_data.get('escalated', False), 'tools_called': response_data.get('tools_called', [])})}\n\n"

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )