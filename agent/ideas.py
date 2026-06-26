from langchain_core.messages import HumanMessage
from core.llm import get_llm

PROMPT_TEMPLATE = """You are a content strategist. Given the topic below, generate exactly 5 distinct article ideas.

Topic: {topic}

Return ONLY a numbered list (1–5), one idea per line. No explanations, no extra text.
Each idea should be a clear, specific article title."""


def generate_ideas(topic: str) -> list[str]:
    llm = get_llm(temperature=0.7)
    prompt = PROMPT_TEMPLATE.format(topic=topic)
    response = llm.invoke([HumanMessage(content=prompt)])
    lines = [l.strip() for l in response.content.strip().splitlines() if l.strip()]
    ideas = []
    for line in lines:
        # strip leading "1. " / "1) " etc.
        if line[0].isdigit():
            line = line.split(".", 1)[-1].split(")", 1)[-1].strip()
        ideas.append(line)
    return ideas[:5]
