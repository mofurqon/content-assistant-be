import asyncio
from typing import Callable

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


async def generate_article(
    idea: str,
    draft: str,
    research_summary: str,
    stream_callback: Callable[[str], None] | None = None,
) -> str:
    llm = get_llm(temperature=0.4)
    prompt = ARTICLE_PROMPT.format(idea=idea, draft=draft, research_summary=research_summary)
    full = ""
    async for chunk in llm.astream([HumanMessage(content=prompt)]):
        if chunk.content:
            full += chunk.content
            if stream_callback:
                stream_callback(chunk.content)
    return full.strip()


def generate_image_prompt(idea: str) -> str:
    llm = get_llm(temperature=0.4)
    response = llm.invoke([HumanMessage(content=IMAGE_PROMPT.format(idea=idea))])
    text = response.content.strip()
    if "**Prompt:**" in text:
        text = text.split("**Prompt:**", 1)[-1].strip()
    return text


async def finalize(idea: str, draft: str, research_summary: str) -> dict:
    article, img_prompt = await asyncio.gather(
        generate_article(idea, draft, research_summary),
        asyncio.to_thread(generate_image_prompt, idea),
    )
    return {"article": article, "image_prompt": img_prompt}
