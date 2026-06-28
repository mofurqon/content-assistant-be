from langchain_core.messages import HumanMessage
from agent.evaluator import evaluate
from core.llm import get_llm
from config.constants import MAX_ITERATIONS, SCORE_THRESHOLD

PROMPT_TEMPLATE = """You are a content editor. Improve the article below based on the evaluation feedback{human_section}.

Article:
{draft}

Evaluation Scores: {scores}
Average Score: {average}/5
Evaluator Reasoning: {reasoning}
{human_block}
Rewrite the article addressing the weaknesses identified. Keep what is already good.
Return only the improved article with no preamble."""


def improve_once(draft: str, eval_result: dict, human_feedback: str = "") -> str:
    """Single-pass rewrite using eval scores and optional human feedback. Returns improved draft."""
    llm = get_llm(temperature=0.4)
    human_section = " and the human reviewer's notes" if human_feedback else ""
    human_block = f"\nHuman Reviewer Notes:\n{human_feedback}\n" if human_feedback else ""
    prompt = PROMPT_TEMPLATE.format(
        draft=draft,
        scores=eval_result["scores"],
        average=eval_result["average"],
        reasoning=eval_result["reasoning"],
        human_section=human_section,
        human_block=human_block,
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()


def improve(draft: str, kb_chunks: list[str], human_feedback: str = "") -> tuple[str, list[dict]]:
    """
    Returns (final_draft, list_of_eval_results).
    Runs up to MAX_ITERATIONS or stops early when average >= SCORE_THRESHOLD.
    human_feedback, when provided, is folded into the first improvement prompt only.
    Used by services/content_pipeline.py for the Streamlit human-review path.
    """
    evals = []
    current = draft

    for i in range(MAX_ITERATIONS):
        result = evaluate(current, kb_chunks)
        evals.append(result)
        if result["average"] >= SCORE_THRESHOLD:
            break
        feedback = human_feedback if i == 0 else ""
        current = improve_once(current, result, feedback)

    if not evals or evals[-1]["average"] < SCORE_THRESHOLD:
        evals.append(evaluate(current, kb_chunks))

    return current, evals
