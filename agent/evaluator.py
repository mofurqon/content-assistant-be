import re
from langchain_core.messages import HumanMessage
from agent.scoring import (
    score_clarity,
    score_completeness,
    score_relevance,
    score_retrieval_relevance,
)
from core.llm import get_embeddings, get_llm
from config.constants import CRITERIA, LLM_JUDGED_CRITERIA

# Retrieval Relevance is always scored deterministically (agent/scoring.py) —
# it uses precomputed kb_scores, no LLM or embedding call needed here.
_LLM_CRITERIA = [c for c in CRITERIA if c in LLM_JUDGED_CRITERIA]

PROMPT_TEMPLATE = """You are a content quality evaluator. Score the article below on each criterion from 1 (poor) to 5 (excellent).

Article:
{draft}

Knowledge Base Context:
{context}

Score each criterion on its own line in EXACTLY this format:
Accuracy: <score>
Actionability: <score>

Then write one paragraph of reasoning explaining the scores."""


def _evaluate_llm_criteria(draft: str, kb_chunks: list[str]) -> tuple[dict[str, int], str]:
    """Accuracy and Actionability need semantic judgement an embedding/heuristic
    proxy can't give: Accuracy requires checking claims against the KB context
    for contradiction, not just topical similarity; Actionability requires
    judging whether guidance is genuinely concrete, not just verb-dense."""
    llm = get_llm(temperature=0.2)
    context = "\n\n---\n\n".join(kb_chunks)
    prompt = PROMPT_TEMPLATE.format(draft=draft, context=context)
    response = llm.invoke([HumanMessage(content=prompt)])
    text = response.content.strip()

    scores = {}
    for criterion in _LLM_CRITERIA:
        match = re.search(rf"{criterion}:\s*([1-5])", text)
        scores[criterion] = int(match.group(1)) if match else 3

    reasoning = re.split(r"\n\s*\n", text)[-1].strip()
    return scores, reasoning


def _evaluate_deterministic_criteria(
    idea: str,
    draft: str,
    draft_vec: list[float] | None,
) -> dict[str, int]:
    return {
        "Clarity": score_clarity(draft),
        "Relevance": score_relevance(idea, draft, draft_vec=draft_vec),
        "Completeness": score_completeness(draft),
    }


def _build_reasoning(deterministic_scores: dict[str, int], llm_reasoning: str) -> str:
    """Templated summary for the deterministic criteria, followed by the LLM's
    own reasoning for Accuracy/Actionability."""
    parts = [f"{name}: {score}/5" for name, score in deterministic_scores.items()]
    return ", ".join(parts) + f". {llm_reasoning}"


def evaluate(idea: str, draft: str, kb_chunks: list[str], kb_scores: list[float]) -> dict:
    embeddings = get_embeddings()
    draft_vec = embeddings.embed_query(draft) if draft.strip() else None

    deterministic_scores = _evaluate_deterministic_criteria(idea, draft, draft_vec)
    llm_scores, llm_reasoning = _evaluate_llm_criteria(draft, kb_chunks)

    scores = {**deterministic_scores, **llm_scores}
    scores["Retrieval Relevance"] = score_retrieval_relevance(kb_scores)

    reasoning = _build_reasoning(deterministic_scores, llm_reasoning)

    avg_score = sum(scores.values()) / len(scores)
    return {"scores": scores, "average": round(avg_score, 2), "reasoning": reasoning}
