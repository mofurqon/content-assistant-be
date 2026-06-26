from langchain_core.messages import HumanMessage
from core.llm import get_llm

PROMPT_TEMPLATE = """You are an expert technical writer. Write a well-structured, informative article based on the idea and knowledge base context below.

Article Idea: {idea}

Knowledge Base Context:
{context}

Requirements:
- Include an introduction, 3–5 main sections with headers, and a conclusion
- Ground your content in the provided context where relevant
- Write in a clear, professional tone
- Aim for 600–900 words

Write the full article now:"""


def generate_draft(idea: str, kb_chunks: list[str]) -> str:
    llm = get_llm(temperature=0.5)
    context = "\n\n---\n\n".join(kb_chunks)
    prompt = PROMPT_TEMPLATE.format(idea=idea, context=context)
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()


def stream_draft(idea: str, kb_chunks: list[str]):
    """
    Yield the draft token-by-token for live UI rendering (st.write_stream).
    st.write_stream returns the full concatenated string when consumed.
    """
    llm = get_llm(temperature=0.5)
    context = "\n\n---\n\n".join(kb_chunks)
    prompt = PROMPT_TEMPLATE.format(idea=idea, context=context)
    for chunk in llm.stream([HumanMessage(content=prompt)]):
        if chunk.content:
            yield chunk.content
