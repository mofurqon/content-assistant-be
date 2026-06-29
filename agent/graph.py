import asyncio
from typing import TypedDict

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt

from agent.evaluator import evaluate
from agent.finalizer import ARTICLE_PROMPT, IMAGE_PROMPT, generate_image_prompt
from agent.generator import PROMPT_TEMPLATE as DRAFT_PROMPT
from agent.ideas import generate_ideas
from agent.improver import improve_once
from agent.researcher import research
from agent.retriever import retrieve_with_scores
from config.constants import MAX_ITERATIONS, SCORE_THRESHOLD
from core.llm import get_llm


class PipelineState(TypedDict):
    # session entry
    topic: str
    ideas: list[str]
    # pipeline data
    idea: str
    kb_chunks: list[str]
    kb_scores: list[float]
    draft: str
    current_draft: str
    evaluations: list[dict]
    iteration: int
    research: dict
    article: str
    image_prompt: str
    # criteria
    human_feedback: str
    target_audience: str
    content_type: str
    tone: str
    # staged resume value written by POST, consumed by GET stream
    pending_resume: str


async def node_generate_ideas(state: PipelineState) -> dict:
    ideas = await asyncio.to_thread(generate_ideas, state["topic"])
    return {"ideas": ideas}


async def node_select_idea(state: PipelineState) -> dict:
    # Pauses here; caller resumes with Command(resume=selected_idea_string)
    selected = interrupt(state["ideas"])
    return {"idea": selected}


async def node_retrieve(state: PipelineState) -> dict:
    pairs = await asyncio.to_thread(retrieve_with_scores, state["idea"])
    return {
        "kb_chunks": [text for text, _ in pairs],
        "kb_scores": [score for _, score in pairs],
    }


async def node_generate(state: PipelineState) -> dict:
    llm = get_llm(temperature=0.5)
    context = "\n\n---\n\n".join(state["kb_chunks"])
    prompt = DRAFT_PROMPT.format(
        idea=state["idea"],
        context=context,
        target_audience=state["target_audience"],
        content_type=state["content_type"],
        tone=state["tone"],
    )
    full = ""
    async for chunk in llm.astream([HumanMessage(content=prompt)]):
        if chunk.content:
            full += chunk.content
    return {"draft": full.strip(), "current_draft": full.strip()}


async def node_review(state: PipelineState) -> dict:
    # Pauses here; caller resumes with Command(resume=human_feedback_string)
    feedback = interrupt(state["current_draft"])
    return {"human_feedback": feedback}


async def node_evaluate(state: PipelineState) -> dict:
    result = await asyncio.to_thread(evaluate, state["current_draft"], state["kb_chunks"], state["kb_scores"])
    return {"evaluations": state["evaluations"] + [result]}


async def node_improve(state: PipelineState) -> dict:
    feedback = state["human_feedback"] if state["iteration"] == 0 else ""
    last_eval = state["evaluations"][-1]
    improved = await asyncio.to_thread(improve_once, state["current_draft"], last_eval, feedback)
    return {"current_draft": improved, "iteration": state["iteration"] + 1}


async def node_research(state: PipelineState) -> dict:
    result = await asyncio.to_thread(research, state["idea"])
    return {"research": result}


async def node_finalize(state: PipelineState) -> dict:
    llm = get_llm(temperature=0.4)
    article_prompt = ARTICLE_PROMPT.format(
        idea=state["idea"],
        draft=state["current_draft"],
        research_summary=state["research"]["summary"],
    )
    full = ""
    async for chunk in llm.astream([HumanMessage(content=article_prompt)]):
        if chunk.content:
            full += chunk.content
    img_prompt = await asyncio.to_thread(generate_image_prompt, state["idea"])
    return {"article": full.strip(), "image_prompt": img_prompt}


def route_after_eval(state: PipelineState) -> str:
    last = state["evaluations"][-1]
    if last["average"] >= SCORE_THRESHOLD or state["iteration"] >= MAX_ITERATIONS:
        return "research"
    return "improve"


_builder = StateGraph(PipelineState)

for _name, _fn in [
    ("generate_ideas", node_generate_ideas),
    ("select_idea", node_select_idea),
    ("retrieve", node_retrieve),
    ("generate", node_generate),
    ("review", node_review),
    ("evaluate", node_evaluate),
    ("improve", node_improve),
    ("research", node_research),
    ("finalize", node_finalize),
]:
    _builder.add_node(_name, _fn)

_builder.set_entry_point("generate_ideas")
_builder.add_edge("generate_ideas", "select_idea")
_builder.add_edge("select_idea", "retrieve")
_builder.add_edge("retrieve", "generate")
_builder.add_edge("generate", "review")
_builder.add_edge("review", "evaluate")
_builder.add_conditional_edges(
    "evaluate",
    route_after_eval,
    {"research": "research", "improve": "improve"},
)
_builder.add_edge("improve", "evaluate")
_builder.add_edge("research", "finalize")
_builder.add_edge("finalize", END)

_checkpointer = MemorySaver()
pipeline_graph = _builder.compile(checkpointer=_checkpointer)
