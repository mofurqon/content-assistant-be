from langchain_core.messages import HumanMessage
from core.llm import get_llm

ARTICLE_PROMPT = """You are a senior technical writer. Combine the improved draft and web research insights below into a polished final article.

Article Idea: {idea}

Improved Draft:
{draft}

Web Research Insights:
{research_summary}

Instructions:
- Seamlessly integrate the research insights into the article where relevant
- Keep the article structure intact (intro, sections, conclusion)
- Do not add a "Sources" section — weave insights naturally into the text
- Aim for 700–1000 words

Write the final article:"""

IMAGE_PROMPT = """Based on the article below, write a detailed image generation prompt for a header image.

Article Title/Idea: {idea}

The prompt should describe a professional, relevant visual in 2–3 sentences. Be specific about style, subject, and mood."""


def stream_article(idea: str, draft: str, research_summary: str):
    """
    Yield the final article token-by-token for live UI rendering
    (st.write_stream returns the full concatenated string when consumed).
    """
    llm = get_llm(temperature=0.4)
    prompt = ARTICLE_PROMPT.format(
        idea=idea,
        draft=draft,
        research_summary=research_summary,
    )
    for chunk in llm.stream([HumanMessage(content=prompt)]):
        if chunk.content:
            yield chunk.content


def generate_image_prompt(idea: str) -> str:
    llm = get_llm(temperature=0.4)
    response = llm.invoke([HumanMessage(content=IMAGE_PROMPT.format(idea=idea))])
    text = response.content.strip()
    if "**Prompt:**" in text:
        text = text.split("**Prompt:**", 1)[-1].strip()
    return text


def finalize(idea: str, draft: str, research_summary: str) -> dict:
    article = "".join(stream_article(idea, draft, research_summary)).strip()
    return {
        "article": article,
        "image_prompt": generate_image_prompt(idea),
    }
