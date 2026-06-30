import json
import uuid

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from agent.graph import pipeline_graph
from api.middleware.rate_limit import rate_limit
from api.schemas.requests import GenerateIdeasRequest
from api.schemas.responses import SessionResponse

router = APIRouter(tags=["ideas"])

_INITIAL_STATE = dict(
    ideas=[],
    idea="",
    kb_chunks=[],
    kb_scores=[],
    draft="",
    current_draft="",
    evaluations=[],
    iteration=0,
    research={},
    article="",
    image_prompt="",
    human_feedback="",
    pending_resume="",
    target_audience="general readers",
    content_type="Article",
    tone="Professional and informative",
)


@router.post("/ideas", response_model=SessionResponse)
async def create_ideas_session(req: GenerateIdeasRequest) -> SessionResponse:
    session_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": session_id}}
    await pipeline_graph.aupdate_state(config, {**_INITIAL_STATE, "topic": req.topic})
    return SessionResponse(session_id=session_id)


@router.get("/ideas/{session_id}/stream")
@rate_limit("10/minute")
async def stream_ideas(request: Request, session_id: str) -> StreamingResponse:
    config = {"configurable": {"thread_id": session_id}}
    snapshot = pipeline_graph.get_state(config)
    if not snapshot or not snapshot.values:
        raise HTTPException(status_code=404, detail="Session not found")

    async def generate():
        async for event in pipeline_graph.astream_events(None, config=config, version="v2"):
            if event["event"] == "on_chat_model_stream":
                token = event["data"]["chunk"].content
                if token:
                    yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
        current = pipeline_graph.get_state(config)
        ideas = current.values.get("ideas", [])
        yield f"data: {json.dumps({'type': 'interrupted', 'ideas': ideas})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
