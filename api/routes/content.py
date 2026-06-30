import json

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from langgraph.types import Command

from agent.graph import pipeline_graph
from api.middleware.rate_limit import rate_limit
from api.schemas.requests import PipelineFeedbackRequest, PipelineStartRequest
from api.schemas.responses import SessionResponse

router = APIRouter(tags=["content"])

_PIPELINE_NODES = {"retrieve", "generate", "review", "evaluate", "improve", "research", "finalize"}


@router.post("/pipeline/{session_id}", response_model=SessionResponse)
async def start_pipeline(session_id: str, req: PipelineStartRequest) -> SessionResponse:
    """Stage selected idea + criteria. Client then GETs /pipeline/{session_id}/stream."""
    config = {"configurable": {"thread_id": session_id}}
    pipeline_graph.update_state(config, {
        "pending_resume": req.selected_idea,
        "target_audience": req.criteria.target_audience,
        "content_type": req.criteria.content_type,
        "tone": req.criteria.tone,
    })
    return SessionResponse(session_id=session_id)


@router.get("/pipeline/{session_id}/stream")
@rate_limit("3/minute")
async def stream_pipeline(request: Request, session_id: str) -> StreamingResponse:
    """Resume from the current interrupt using the staged pending_resume value.
    Emits awaiting_feedback after the draft, or done after the final article."""
    config = {"configurable": {"thread_id": session_id}}

    snapshot = pipeline_graph.get_state(config)
    resume_value = snapshot.values.get("pending_resume", "")
    pipeline_graph.update_state(config, {"pending_resume": ""})

    async def generate():
        async for event in pipeline_graph.astream_events(
            Command(resume=resume_value), config=config, version="v2"
        ):
            kind = event["event"]
            if kind == "on_chat_model_stream":
                token = event["data"]["chunk"].content
                if token:
                    yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
            elif kind == "on_chain_start":
                node = event.get("metadata", {}).get("langgraph_node")
                if node in _PIPELINE_NODES:
                    yield f"data: {json.dumps({'type': 'node', 'name': node})}\n\n"

        final_snapshot = pipeline_graph.get_state(config)
        if final_snapshot.next:
            # Still paused — must be at the review interrupt after draft generation
            draft = final_snapshot.values.get("draft", "")
            yield f"data: {json.dumps({'type': 'awaiting_feedback', 'draft': draft})}\n\n"
        else:
            yield f"data: {json.dumps({'type': 'done', 'result': _serialize_final(final_snapshot.values)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.post("/pipeline/{session_id}/resume", response_model=SessionResponse)
async def resume_pipeline(session_id: str, req: PipelineFeedbackRequest) -> SessionResponse:
    """Stage human feedback. Client then GETs /pipeline/{session_id}/stream again."""
    config = {"configurable": {"thread_id": session_id}}
    pipeline_graph.update_state(config, {
        "pending_resume": req.human_feedback,
        "human_feedback": req.human_feedback,
    })
    return SessionResponse(session_id=session_id)


def _serialize_final(state: dict) -> dict:
    evals = state.get("evaluations", [])
    research = state.get("research", {})
    return {
        "idea": state.get("idea", ""),
        "draft": state.get("draft", ""),
        "article": state.get("article", ""),
        "image_prompt": state.get("image_prompt", ""),
        "evaluations": evals,
        "research": {
            "queries": research.get("queries", []),
            "summary": research.get("summary", ""),
        },
    }
