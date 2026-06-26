from agent.ideas import generate_ideas
from agent.retriever import retrieve
from agent.generator import generate_draft
from agent.improver import improve
from agent.researcher import research
from agent.finalizer import finalize


def run_pipeline(topic: str, selected_idea: str, status_cb=None) -> dict:
    """
    Runs the full content generation pipeline.

    Args:
        topic: The raw user-supplied topic.
        selected_idea: The idea chosen by the user from generate_ideas().
        status_cb: Optional callable(step: str, data: any) for streaming status to UI.

    Returns a dict with keys: idea, kb_chunks, draft, evals, research, article, image_prompt.
    """
    def _emit(step, data=None):
        if status_cb:
            status_cb(step, data)

    _emit("retrieval")
    kb_chunks = retrieve(selected_idea)

    _emit("generation")
    draft = generate_draft(selected_idea, kb_chunks)

    _emit("improvement")
    improved_draft, evals = improve(draft, kb_chunks)

    _emit("research")
    research_result = research(selected_idea)

    _emit("finalization")
    final = finalize(selected_idea, improved_draft, research_result["summary"])

    return {
        "idea": selected_idea,
        "kb_chunks": kb_chunks,
        "draft": draft,
        "evals": evals,
        "research": research_result,
        "article": final["article"],
        "image_prompt": final["image_prompt"],
    }
