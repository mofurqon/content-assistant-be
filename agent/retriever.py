import logging

from langchain_chroma import Chroma
from core.llm import get_embeddings
from config.constants import CHROMA_PATH, COLLECTION_NAME, KB_SIMILARITY_THRESHOLD, TOP_K

_log = logging.getLogger("retriever")


def _get_vectorstore():
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_PATH),
    )


def _log_scores(results: list, query: str) -> None:
    _log.info("KB retrieval — query=%r top_k=%d threshold=%.2f", query, TOP_K, KB_SIMILARITY_THRESHOLD)
    for i, (doc, score) in enumerate(results, start=1):
        passed = score >= KB_SIMILARITY_THRESHOLD
        _log.info(
            "  [%d/%d] score=%.4f %s | preview=%r",
            i, len(results), score, "PASS" if passed else "SKIP",
            doc.page_content[:80].replace("\n", " "),
        )
    kept = sum(1 for _, s in results if s >= KB_SIMILARITY_THRESHOLD)
    _log.info("  kept %d/%d chunks", kept, len(results))


def _filter_or_fallback(results: list) -> list:
    """Return chunks above threshold. If none qualify, fall back to all top-K so the LLM always has context."""
    passing = [(doc, score) for doc, score in results if score >= KB_SIMILARITY_THRESHOLD]
    if passing:
        return passing
    _log.warning(
        "No chunks met threshold=%.2f — falling back to all %d top-K results",
        KB_SIMILARITY_THRESHOLD, len(results),
    )
    return results


def retrieve(query: str) -> list[str]:
    """Returns chunk texts above KB_SIMILARITY_THRESHOLD, falling back to top-K if none qualify."""
    results = _get_vectorstore().similarity_search_with_relevance_scores(query, k=TOP_K)
    _log_scores(results, query)
    return [doc.page_content for doc, _ in _filter_or_fallback(results)]


def retrieve_with_scores(query: str) -> list[tuple[str, float]]:
    """Returns (text, score) pairs above KB_SIMILARITY_THRESHOLD, falling back to top-K if none qualify."""
    results = _get_vectorstore().similarity_search_with_relevance_scores(query, k=TOP_K)
    _log_scores(results, query)
    return [(doc.page_content, round(score, 4)) for doc, score in _filter_or_fallback(results)]
