import asyncio
import os

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from config.constants import FETCH_MAX_CHARS, FIRECRAWL_CONCURRENCY, FIRECRAWL_LIMIT
from core.llm import get_llm

_sem: asyncio.Semaphore | None = None


def _get_sem() -> asyncio.Semaphore:
    global _sem
    if _sem is None:
        _sem = asyncio.Semaphore(FIRECRAWL_CONCURRENCY)
    return _sem


QUERY_PROMPT = """Given the article idea below, generate 3 concise web search queries to find supporting facts, statistics, or expert opinions.

Article Idea: {idea}

Return ONLY a numbered list of 3 queries, one per line."""

SUMMARIZE_PROMPT = """Summarize the key insights from the web content below that are relevant to the article idea.

Article Idea: {idea}

Web Content:
{content}

Write a concise summary (3–5 bullet points) of the most relevant facts or insights."""


def _generate_queries(idea: str, llm: ChatGoogleGenerativeAI) -> list[str]:
    response = llm.invoke([HumanMessage(content=QUERY_PROMPT.format(idea=idea))])
    lines = [l.strip() for l in response.content.strip().splitlines() if l.strip()]
    queries = []
    for line in lines:
        if line[0].isdigit():
            line = line.split(".", 1)[-1].split(")", 1)[-1].strip()
        queries.append(line)
    return queries[:3]


async def _fetch_content(query: str) -> str:
    firecrawl_key = os.environ.get("FIRECRAWL_API_KEY", "")
    if not firecrawl_key:
        return f"[Mock result for: {query}] No live data — FIRECRAWL_API_KEY not set."

    async with _get_sem():
        try:
            from firecrawl import AsyncFirecrawlApp
            app = AsyncFirecrawlApp(api_key=firecrawl_key)
            result = await app.search(query, limit=FIRECRAWL_LIMIT)
            texts = []
            for item in result.data:
                texts.append(item.get("markdown") or item.get("content") or "")
            return "\n\n".join(texts)[:FETCH_MAX_CHARS]
        except Exception as e:
            return f"[Firecrawl error: {e}]"


async def research(idea: str) -> dict:
    llm = get_llm(temperature=0.3)
    queries = _generate_queries(idea, llm)
    contents = await asyncio.gather(*[_fetch_content(q) for q in queries])
    raw_results = dict(zip(queries, contents))

    combined = "\n\n".join(f"Query: {q}\n{c}" for q, c in raw_results.items())
    summary_prompt = SUMMARIZE_PROMPT.format(idea=idea, content=combined)
    summary_response = llm.invoke([HumanMessage(content=summary_prompt)])

    return {
        "queries": queries,
        "summary": summary_response.content.strip(),
    }
