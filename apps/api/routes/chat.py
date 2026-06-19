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
    conversation_history: list = []
    intent_override: str = None
    transaction_id: str = None


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

        # Let the user know we started
        yield f"data: {json.dumps({'type': 'status', 'content': 'Checking memory...'})}\n\n"

        try:
            final_state = None
            
            # Iterate through the actual LangGraph state updates as nodes finish
            async for update in pipeline.process_stream(
                message=request.message,
                session_id=session_id,
                user_id=user_id,
                conversation_history=request.conversation_history,
                intent_override=request.intent_override,
                transaction_id=request.transaction_id
            ):
                # 'update' is a dict where key is the node name and value is the state update
                for node_name, state_update in update.items():
                    final_state = state_update  # Keep track of the latest state
                    
                    # Map the internal node names to user-friendly status messages
                    status_messages = {
                        "greeting": "Looking up user history...",
                        "intent_classification": "Analyzing intent...",
                        "knowledge_retrieval": "Searching knowledge base...",
                        "action_execution": "Executing requested actions...",
                        "resolution_execution": "Building resolution steps...",
                        "quality_check": "Evaluating AI response quality...",
                        "revision": "Revising response based on judge feedback...",
                        "escalation": "Preparing human handoff...",
                        "collecting_info": "Determining missing information...",
                        "response_delivery": "Finalizing response..."
                    }
                    
                    if node_name in status_messages:
                        yield f"data: {json.dumps({'type': 'status', 'content': status_messages[node_name]})}\n\n"

            # Check if the graph was interrupted (paused at collecting_info)
            from core.orchestrator.pipeline import app
            config = {"configurable": {"thread_id": session_id}}
            graph_state = await app.aget_state(config)
            
            if graph_state and graph_state.next:
                # Graph is paused — the interrupt value is the follow-up question
                interrupt_question = graph_state.tasks[0].interrupts[0].value if graph_state.tasks else ""
                
                yield f"data: {json.dumps({'type': 'token', 'content': interrupt_question})}\n\n"
                
                meta = {
                    "agent_used": "collecting_info",
                    "escalated": False,
                    "tools_called": [],
                    "quality_scores": None,
                    "quality_passed": None,
                    "revision_count": 0,
                    "total_tokens": 0,
                }
                yield f"data: {json.dumps({'type': 'metadata', **meta})}\n\n"
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                return

            # Get the full final state from LangGraph
            final_full_state = graph_state.values if graph_state else {}

            if not final_full_state:
                raise Exception("State machine returned no output")

            # The final response is now fully generated and approved by the judge
            full_response = final_full_state.get("agent_response", "")
            if full_response:
                full_response = await guardrails.sanitize_output(full_response)

            yield f"data: {json.dumps({'type': 'token', 'content': full_response})}\n\n"

            # Send the metadata
            meta = {
                "agent_used": final_full_state.get("active_agent", "unknown"),
                "escalated": final_full_state.get("escalated", False),
                "tools_called": final_full_state.get("tools_called", []),
                "quality_scores": final_full_state.get("quality_scores"),
                "quality_passed": final_full_state.get("quality_passed"),
                "revision_count": final_full_state.get("revision_count", 0),
                "total_tokens": final_full_state.get("total_tokens", 0),
            }
            yield f"data: {json.dumps({'type': 'metadata', **meta})}\n\n"

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            import traceback
            traceback.print_exc()
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


class EndConversationRequest(BaseModel):
    user_id: str
    conversation_history: list
    intent: str = "general_inquiry"

@router.post("/end-conversation")
async def end_conversation(request: EndConversationRequest):
    """
    Summarizes the chat and saves it as an episodic memory to the SupportTickets table.
    """
    from infrastructure.llm.nim_client import nim_client
    from infrastructure.db.connection import AsyncSessionLocal
    from infrastructure.db.models import SupportTicket
    
    # Generate summary
    transcript = ""
    for msg in request.conversation_history:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        transcript += f"{role.upper()}: {content}\n"
        
    response = await nim_client.complete(
        messages=[
            {"role": "system", "content": "You summarize customer support chats into one short sentence explaining the problem and resolution. Return a raw JSON object with 'problem' and 'resolution' keys."},
            {"role": "user", "content": transcript}
        ],
        tier="fast",
        max_tokens=100
    )
    
    summary_text = response.get("content", '{"problem": "Unknown", "resolution": "Ended"}')
    
    async with AsyncSessionLocal() as session:
        ticket = SupportTicket(
            user_id=request.user_id,
            intent=request.intent,
            summary=summary_text,
            status="resolved"
        )
        session.add(ticket)
        await session.commit()
        
    return {"status": "success", "ticket_id": ticket.id}