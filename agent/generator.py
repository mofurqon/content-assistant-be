from langchain_core.messages import HumanMessage
from core.llm import get_llm

PROMPT_TEMPLATE = """You are an expert technical writer. Write a well-structured, informative article based on the idea and knowledge base context below.

Article Idea: {idea}

Knowledge Base Context:
{context}

Criteria:
Target Audience: {target_audience}
Content Type: {content_type}
Tone: {tone}

Requirements:
- Include an introduction, 3–5 main sections with headers, and a conclusion
- Ground your content in the provided context where relevant
- Tailor the language and depth to the specified target audience
- Match the specified tone throughout
- Aim for 600–900 words

Write the full article now:"""


def generate_draft(
    idea: str,
    kb_chunks: list[str],
    target_audience: str = "general readers",
    content_type: str = "Article",
    tone: str = "Professional and informative",
) -> str:
    llm = get_llm(temperature=0.5)
    context = "\n\n---\n\n".join(kb_chunks)
    prompt = PROMPT_TEMPLATE.format(
        idea=idea,
        context=context,
        target_audience=target_audience,
        content_type=content_type,
        tone=tone,
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()


def stream_draft(
    idea: str,
    kb_chunks: list[str],
    target_audience: str = "general readers",
    content_type: str = "Article",
    tone: str = "Professional and informative",
):
    llm = get_llm(temperature=0.5)
    context = "\n\n---\n\n".join(kb_chunks)
    prompt = PROMPT_TEMPLATE.format(
        idea=idea,
        context=context,
        target_audience=target_audience,
        content_type=content_type,
        tone=tone,
    )
    for chunk in llm.stream([HumanMessage(content=prompt)]):
        if chunk.content:
            yield chunk.content
