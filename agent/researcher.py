import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from core.llm import get_llm
from config.constants import FIRECRAWL_LIMIT, FETCH_MAX_CHARS

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


def _fetch_content(query: str) -> str:
    firecrawl_key = os.environ.get("FIRECRAWL_API_KEY", "")
    if not firecrawl_key:
        return f"[Mock result for: {query}] No live data — FIRECRAWL_API_KEY not set."

    try:
        from firecrawl import FirecrawlApp
        app = FirecrawlApp(api_key=firecrawl_key)
        result = app.search(query, limit=FIRECRAWL_LIMIT)
        texts = []
        for item in result.get("data", []):
            texts.append(item.get("markdown") or item.get("content") or "")
        return "\n\n".join(texts)[:FETCH_MAX_CHARS]
    except Exception as e:
        return f"[Firecrawl error: {e}]"


def research(idea: str) -> dict:
    llm = get_llm(temperature=0.3)
    queries = _generate_queries(idea, llm)
    raw_results = {q: _fetch_content(q) for q in queries}

    combined = "\n\n".join(f"Query: {q}\n{c}" for q, c in raw_results.items())
    summary_prompt = SUMMARIZE_PROMPT.format(idea=idea, content=combined)
    summary_response = llm.invoke([HumanMessage(content=summary_prompt)])

    return {
        "queries": queries,
        "summary": summary_response.content.strip(),
    }
