"""Deterministic scoring functions for evaluation criteria that don't need
LLM judgement. Each function returns an int score 1-5 from an objective
signal (embedding cosine similarity), normalized via _normalize().

Normalization bounds (config/constants.py: RETRIEVAL_RELEVANCE_*, CLARITY_*,
RELEVANCE_*, COMPLETENESS_*) are placeholders until calibrated against real
data — see scripts/calibrate_threshold.py for the calibration pattern (it
currently calibrates KB_SIMILARITY_THRESHOLD only; RETRIEVAL_RELEVANCE_LOW
reuses its output). Don't trust the defaults; they exist so the pipeline
runs, not because they're correct.

Accuracy and Actionability are judged by the LLM instead (agent/evaluator.py,
config/constants.py:LLM_JUDGED_CRITERIA) — no deterministic scorer here.
"""

import logging
import math
import re

import textstat

from config.constants import (
    CLARITY_HIGH,
    CLARITY_LOW,
    COMPLETENESS_HIGH,
    COMPLETENESS_LOW,
    RELEVANCE_HIGH,
    RELEVANCE_LOW,
    RETRIEVAL_RELEVANCE_HIGH,
    RETRIEVAL_RELEVANCE_LOW,
    TARGET_WORD_RANGE,
)
from core.llm import get_embeddings

_log = logging.getLogger("scoring")


def _normalize(value: float, low: float, high: float) -> int:
    """Map a raw value onto an integer 1-5 band. value==low -> 1, value==high -> 5.
    Values outside [low, high] are clamped, not extrapolated."""
    if high <= low:
        raise ValueError(f"high ({high}) must be greater than low ({low})")
    ratio = (value - low) / (high - low)
    return max(1, min(5, round(1 + ratio * 4)))


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def score_retrieval_relevance(kb_scores: list[float]) -> int:
    """How relevant were the retrieved chunks to the query (idea)?

    Raw signal: cosine similarity scores from agent/retriever.py, used as-is
    — NOT filtered by KB_SIMILARITY_THRESHOLD. Filtering first and then
    normalizing over [threshold, 1.0] creates a cliff: a near-miss fallback
    set (e.g. all chunks just under threshold) would score identically to a
    set of irrelevant chunks. Normalizing the raw average preserves that
    distinction.

    NOTE: this measures retrieval quality only — whether good chunks were
    *found* — not whether the generated draft actually used them (Accuracy
    is LLM-judged, see agent/evaluator.py).
    """
    if not kb_scores:
        return 1
    avg = sum(kb_scores) / len(kb_scores)
    return _normalize(avg, RETRIEVAL_RELEVANCE_LOW, RETRIEVAL_RELEVANCE_HIGH)


def score_clarity(draft: str) -> int:
    """Flesch Reading Ease: higher = easier to read. Range is technically
    unbounded but real-world prose mostly falls 0-100."""
    if not draft.strip():
        return 1
    fre = textstat.flesch_reading_ease(draft)
    return _normalize(fre, CLARITY_LOW, CLARITY_HIGH)


def score_relevance(
    idea: str,
    draft: str,
    idea_vec: list[float] | None = None,
    draft_vec: list[float] | None = None,
) -> int:
    """Cosine similarity between the article idea and the draft as a whole.

    idea_vec / draft_vec: pass precomputed embeddings to avoid re-embedding
    text evaluate() may already have embedded elsewhere. If not provided,
    embeds them here.
    """
    if not draft.strip():
        return 1
    embeddings = get_embeddings()
    if idea_vec is None:
        idea_vec = embeddings.embed_query(idea)
    if draft_vec is None:
        draft_vec = embeddings.embed_query(draft)
    sim = _cosine_similarity(idea_vec, draft_vec)
    _log.info(
        "Relevance: raw_sim=%.4f bounds=[%.4f, %.4f]",
        sim, RELEVANCE_LOW, RELEVANCE_HIGH,
    )
    return _normalize(sim, RELEVANCE_LOW, RELEVANCE_HIGH)


def score_completeness(draft: str, target_word_range: tuple[int, int] = TARGET_WORD_RANGE) -> int:
    """Blends word-count fit against the generator's target range with a
    section-header count (the generator prompt asks for 3-5 headed
    sections). Purely structural — says nothing about depth of content."""
    if not draft.strip():
        return 1

    word_count = len(draft.split())
    low, high = target_word_range
    mid = (low + high) / 2
    half_width = (high - low) / 2
    word_fit = max(0.0, 1.0 - abs(word_count - mid) / (half_width * 2))

    header_count = len(re.findall(r"^#{1,3}\s+\S|^[A-Z][\w\s]{3,60}:?\s*$", draft, re.MULTILINE))
    header_fit = min(header_count / 3, 1.0)  # 3+ headers = full credit

    raw = word_fit * 0.6 + header_fit * 0.4
    return _normalize(raw, COMPLETENESS_LOW, COMPLETENESS_HIGH)
