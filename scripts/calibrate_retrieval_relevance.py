"""
One-time offline script to find realistic bounds for score_retrieval_relevance()
(agent/scoring.py), replacing the guessed RETRIEVAL_RELEVANCE_LOW/HIGH in
constants.py.

RETRIEVAL_RELEVANCE_LOW/HIGH were seeded from calibrate_threshold.py's F1-optimal
binary cutoff (relevant vs. irrelevant question/chunk pairs) — a different
statistic than what score_retrieval_relevance() actually consumes: the AVERAGE
cosine similarity of the top-K chunks retriever.py returns for one query. This
script measures that real signal directly across a batch of realistic queries
and recommends bounds from the observed distribution.

Requires the KB to already be ingested with collection_metadata={"hnsw:space":
"cosine"} (see ingest/ingest.py) so retriever.py's relevance scores are true
cosine similarity.

Usage:
    python scripts/calibrate_retrieval_relevance.py

After running, set RETRIEVAL_RELEVANCE_LOW / RETRIEVAL_RELEVANCE_HIGH in
config/constants.py to the recommended values printed below.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Allow importing from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.retriever import retrieve_with_scores
from config.constants import COLLECTION_NAME, TOP_K
from config.settings import require_env

# Realistic idea-like queries: mostly KB-domain (software testing) phrasing,
# plus a couple of off-KB queries, so the calibration reflects the real spread
# of queries the pipeline actually retrieves against — not just one topic.
QUERIES = [
    "Why Shift-Left Testing Reduces Production Bugs",
    "A Practical Guide to the Test Pyramid",
    "Unit vs Integration vs E2E: Choosing the Right Test",
    "Test-Driven Development in Practice",
    "How to Build a Reliable CI Pipeline",
    "Code Review Best Practices for Engineering Teams",
    "Cloud Cost Optimization Strategies",
    "Remote Team Productivity Tips",
]

LOW_PERCENTILE = 10
HIGH_PERCENTILE = 90


def build_samples() -> list[dict]:
    samples = []
    for i, query in enumerate(QUERIES):
        print(f"[{i+1}/{len(QUERIES)}] {query} ...", end=" ", flush=True)
        results = retrieve_with_scores(query)
        scores = [s for _, s in results]
        avg = sum(scores) / len(scores) if scores else 0.0
        print(f"n={len(scores)} avg={avg:.4f}")
        samples.append({
            "query": query,
            "scores": scores,
            "avg": round(avg, 4),
        })
    return samples


def print_table(samples: list[dict]) -> None:
    print("\nQuery | N | Avg Score")
    print("------|---|----------")
    for s in samples:
        print(f"{s['query'][:40]:40s} | {len(s['scores']):1d} | {s['avg']:.4f}")


def recommend_bounds(samples: list[dict]) -> tuple[float, float]:
    avgs = np.array([s["avg"] for s in samples])
    low = float(np.percentile(avgs, LOW_PERCENTILE))
    high = float(np.percentile(avgs, HIGH_PERCENTILE))
    return low, high


def save_result(samples: list[dict], low: float, high: float) -> None:
    result = {
        "recommended_retrieval_relevance_low": round(low, 4),
        "recommended_retrieval_relevance_high": round(high, 4),
        "low_percentile": LOW_PERCENTILE,
        "high_percentile": HIGH_PERCENTILE,
        "top_k": TOP_K,
        "collection": COLLECTION_NAME,
        "model": require_env("GEMINI_EMBED_MODEL"),
        "samples": samples,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    out_path = Path(__file__).parent / "calibration_result_retrieval_relevance.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nResult saved to {out_path}")


def main() -> None:
    print("=== Retrieval Relevance Calibration ===")
    samples = build_samples()

    print_table(samples)

    low, high = recommend_bounds(samples)
    print(f"\nRecommended RETRIEVAL_RELEVANCE_LOW  = {low:.4f}  (p{LOW_PERCENTILE})")
    print(f"Recommended RETRIEVAL_RELEVANCE_HIGH = {high:.4f}  (p{HIGH_PERCENTILE})")
    print("\nUpdate RETRIEVAL_RELEVANCE_LOW / RETRIEVAL_RELEVANCE_HIGH in config/constants.py")

    save_result(samples, low, high)


if __name__ == "__main__":
    main()
