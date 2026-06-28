import re
from langchain_core.messages import HumanMessage
from core.llm import get_llm
from config.constants import CRITERIA, KB_SIMILARITY_THRESHOLD

# KB Alignment is scored deterministically from retrieval scores, not by the LLM
_LLM_CRITERIA = [c for c in CRITERIA if c != "KB Alignment"]

PROMPT_TEMPLATE = """You are a content quality evaluator. Score the article below on each criterion from 1 (poor) to 5 (excellent).

Article:
{draft}

Knowledge Base Context:
{context}

Score each criterion on its own line in EXACTLY this format:
Clarity: <score>
Relevance: <score>
Completeness: <score>
Accuracy: <score>
Actionability: <score>

Then write one paragraph of reasoning explaining the scores."""


def evaluate(draft: str, kb_chunks: list[str], kb_scores: list[float]) -> dict:
    llm = get_llm(temperature=0.2)
    context = "\n\n---\n\n".join(kb_chunks)
    prompt = PROMPT_TEMPLATE.format(draft=draft, context=context)
    response = llm.invoke([HumanMessage(content=prompt)])
    text = response.content.strip()

    scores = {}
    for criterion in _LLM_CRITERIA:
        match = re.search(rf"{criterion}:\s*([1-5])", text)
        scores[criterion] = int(match.group(1)) if match else 3

    # KB Alignment: derived from cosine similarity scores, not LLM judgement
    passing = [s for s in kb_scores if s >= KB_SIMILARITY_THRESHOLD]
    if not passing:
        scores["KB Alignment"] = 1
    else:
        avg = sum(passing) / len(passing)
        scores["KB Alignment"] = max(1, min(5, round(1 + avg * 4)))

    avg_score = sum(scores.values()) / len(scores)
    reasoning = re.split(r"\n\s*\n", text)[-1].strip()

    return {"scores": scores, "average": round(avg_score, 2), "reasoning": reasoning}
