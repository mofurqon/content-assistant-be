from langchain_core.messages import HumanMessage
from agent.evaluator import evaluate
from core.llm import get_llm
from config.constants import MAX_ITERATIONS, SCORE_THRESHOLD

PROMPT_TEMPLATE = """You are a content editor. Improve the article below based on the evaluation feedback.

Article:
{draft}

Evaluation Scores: {scores}
Average Score: {average}/5
Evaluator Reasoning: {reasoning}

Rewrite the article addressing the weaknesses identified. Keep what is already good.
Return only the improved article with no preamble."""


def improve(draft: str, kb_chunks: list[str]) -> tuple[str, list[dict]]:
    """
    Returns (final_draft, list_of_eval_results).
    Runs up to MAX_ITERATIONS or stops early when average >= SCORE_THRESHOLD.
    """
    llm = get_llm(temperature=0.4)
    evals = []
    current = draft

    for _ in range(MAX_ITERATIONS):
        result = evaluate(current, kb_chunks)
        evals.append(result)

        if result["average"] >= SCORE_THRESHOLD:
            break

        prompt = PROMPT_TEMPLATE.format(
            draft=current,
            scores=result["scores"],
            average=result["average"],
            reasoning=result["reasoning"],
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        current = response.content.strip()

    # Final eval if we exited the loop by hitting MAX_ITERATIONS without passing
    if not evals or evals[-1]["average"] < SCORE_THRESHOLD:
        evals.append(evaluate(current, kb_chunks))

    return current, evals
