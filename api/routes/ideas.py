import json
import uuid

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from agent.graph import pipeline_graph
from api.schemas.requests import GenerateIdeasRequest
from api.schemas.responses import SessionResponse

router = APIRouter(tags=["ideas"])

# Holds topic per session until GET /ideas/{session_id}/stream consumes it.
# MemorySaver is also in-process, so this shares the same lifetime.
_pending: dict[str, str] = {}

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
    _pending[session_id] = req.topic
    return SessionResponse(session_id=session_id)


@router.get("/ideas/{session_id}/stream")
async def stream_ideas(session_id: str) -> StreamingResponse:
    topic = _pending.pop(session_id, None)
    if topic is None:
        raise HTTPException(status_code=404, detail="Session not found or already streamed")

    config = {"configurable": {"thread_id": session_id}}
    initial_state = {**_INITIAL_STATE, "topic": topic}

    async def generate():
        async for event in pipeline_graph.astream_events(initial_state, config=config, version="v2"):
            if event["event"] == "on_chat_model_stream":
                token = event["data"]["chunk"].content
                if token:
                    yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
        snapshot = pipeline_graph.get_state(config)
        ideas = snapshot.values.get("ideas", [])
        yield f"data: {json.dumps({'type': 'interrupted', 'ideas': ideas})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
