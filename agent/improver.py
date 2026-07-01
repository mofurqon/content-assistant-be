from langchain_core.messages import HumanMessage
from core.llm import get_llm

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
