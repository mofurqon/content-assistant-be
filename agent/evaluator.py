import re
from langchain_core.messages import HumanMessage
from core.llm import get_llm
from config.constants import CRITERIA

PROMPT_TEMPLATE = """You are a content quality evaluator. Score the article below on each criterion from 1 (poor) to 5 (excellent).

Article:
{draft}

Knowledge Base Context (for KB Alignment):
{context}

Score each criterion on its own line in EXACTLY this format:
Clarity: <score>
Relevance: <score>
Completeness: <score>
Accuracy: <score>
Actionability: <score>
KB Alignment: <score>

Then write one paragraph of reasoning explaining the scores."""


def evaluate(draft: str, kb_chunks: list[str]) -> dict:
    llm = get_llm(temperature=0.2)
    context = "\n\n---\n\n".join(kb_chunks)
    prompt = PROMPT_TEMPLATE.format(draft=draft, context=context)
    response = llm.invoke([HumanMessage(content=prompt)])
    text = response.content.strip()

    scores = {}
    for criterion in CRITERIA:
        match = re.search(rf"{criterion}:\s*([1-5])", text)
        scores[criterion] = int(match.group(1)) if match else 3

    avg = sum(scores.values()) / len(scores)
    reasoning = re.split(r"\n\s*\n", text)[-1].strip()

    return {"scores": scores, "average": round(avg, 2), "reasoning": reasoning}
