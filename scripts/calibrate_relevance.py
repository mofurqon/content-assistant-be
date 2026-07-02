"""
One-time offline script to find realistic bounds for score_relevance()
(agent/scoring.py), replacing the guessed RELEVANCE_LOW/HIGH in constants.py.

score_relevance() measures cosine similarity between the article idea and the
generated draft as a whole (did the draft stay on-topic with the idea it was
given) — a different signal than Retrieval Relevance (query vs. retrieved KB
chunks). This script generates real drafts through the actual pipeline steps
(ideas -> retrieve -> generate), computes that same idea-vs-draft cosine
similarity, and recommends bounds from the observed distribution.

Usage:
    python scripts/calibrate_relevance.py

After running, set RELEVANCE_LOW / RELEVANCE_HIGH in config/constants.py to
the recommended values printed below.
"""

import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Allow importing from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.generator import generate_draft
from agent.ideas import generate_ideas
from agent.retriever import retrieve
from agent.scoring import _cosine_similarity
from config.settings import require_env
from core.llm import get_embeddings

# Mostly KB-domain topics (what the pipeline is actually tuned for) plus a
# couple of off-KB topics, so the calibration reflects the real spread of
# input rather than one narrow subject. Same list as calibrate_clarity.py so
# results are directly comparable across the two calibration runs.
TOPICS = [
    "software testing best practices",
    "test-driven development",
    "continuous integration and deployment",
    "unit testing vs integration testing",
    "automated regression testing",
    "code review practices",
    "cloud cost optimization",
    "remote team productivity",
]

LOW_PERCENTILE = 10
HIGH_PERCENTILE = 90


async def build_draft(topic: str) -> tuple[str, str]:
    ideas = generate_ideas(topic)
    idea = ideas[0]
    kb_chunks = retrieve(idea)
    time.sleep(4)  # ~15 RPM free-tier limit for Gemini Flash, between ideas call and draft call
    draft = await generate_draft(idea, kb_chunks)
    return idea, draft


async def build_samples() -> list[dict]:
    embeddings = get_embeddings()
    samples = []
    for i, topic in enumerate(TOPICS):
        if i > 0:
            time.sleep(4)  # ~15 RPM free-tier limit for Gemini Flash, between topics
        print(f"[{i+1}/{len(TOPICS)}] {topic} ...", end=" ", flush=True)
        idea, draft = await build_draft(topic)
        idea_vec = embeddings.embed_query(idea)
        draft_vec = embeddings.embed_query(draft)
        sim = _cosine_similarity(idea_vec, draft_vec)
        print(f"sim={sim:.4f}")
        samples.append({
            "topic": topic,
            "idea": idea,
            "word_count": len(draft.split()),
            "similarity": round(sim, 4),
        })
    return samples


def print_table(samples: list[dict]) -> None:
    print("\nTopic | Words | Idea-Draft Similarity")
    print("------|-------|----------------------")
    for s in samples:
        print(f"{s['topic'][:40]:40s} | {s['word_count']:5d} | {s['similarity']:.4f}")


def recommend_bounds(samples: list[dict]) -> tuple[float, float]:
    sims = np.array([s["similarity"] for s in samples])
    low = float(np.percentile(sims, LOW_PERCENTILE))
    high = float(np.percentile(sims, HIGH_PERCENTILE))
    return low, high


def save_result(samples: list[dict], low: float, high: float) -> None:
    result = {
        "recommended_relevance_low": round(low, 4),
        "recommended_relevance_high": round(high, 4),
        "low_percentile": LOW_PERCENTILE,
        "high_percentile": HIGH_PERCENTILE,
        "model": require_env("GEMINI_EMBED_MODEL"),
        "samples": samples,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    out_path = Path(__file__).parent / "calibration_result_relevance.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nResult saved to {out_path}")


def main() -> None:
    print("=== Relevance (Idea-Draft Cosine Similarity) Calibration ===")
    samples = asyncio.run(build_samples())

    print_table(samples)

    low, high = recommend_bounds(samples)
    print(f"\nRecommended RELEVANCE_LOW  = {low:.4f}  (p{LOW_PERCENTILE})")
    print(f"Recommended RELEVANCE_HIGH = {high:.4f}  (p{HIGH_PERCENTILE})")
    print("\nUpdate RELEVANCE_LOW / RELEVANCE_HIGH in config/constants.py")

    save_result(samples, low, high)


if __name__ == "__main__":
    main()
