from langchain_chroma import Chroma
from core.llm import get_embeddings
from config.constants import CHROMA_PATH, COLLECTION_NAME, TOP_K


def _get_vectorstore():
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_PATH),
    )


def retrieve(query: str) -> list[str]:
    docs = _get_vectorstore().similarity_search(query, k=TOP_K)
    return [doc.page_content for doc in docs]


def retrieve_with_scores(query: str) -> list[tuple[str, float]]:
    """Returns (text, relevance_score) pairs. Score is in [0, 1], higher = more similar."""
    results = _get_vectorstore().similarity_search_with_relevance_scores(query, k=TOP_K)
    return [(doc.page_content, round(score, 4)) for doc, score in results]
