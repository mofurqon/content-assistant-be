"""
One-time offline script to find realistic bounds for score_completeness()
(agent/scoring.py), replacing the guessed COMPLETENESS_LOW/HIGH in
constants.py.

score_completeness() blends word-count fit against TARGET_WORD_RANGE with a
section-header count into one raw score in [0, 1] (word_fit * 0.6 + header_fit
* 0.4) — purely structural, no embeddings or KB involved. This script
generates real drafts through the actual pipeline steps (ideas -> retrieve ->
generate), computes that same raw signal, and recommends bounds from the
observed distribution.

Usage:
    python scripts/calibrate_completeness.py

After running, set COMPLETENESS_LOW / COMPLETENESS_HIGH in
config/constants.py to the recommended values printed below.
"""

import asyncio
import json
import re
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
from config.constants import TARGET_WORD_RANGE
from config.settings import require_env

# Same 8 topics as calibrate_clarity.py / calibrate_relevance.py so results
# are directly comparable across calibration runs.
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


def _raw_completeness(draft: str, target_word_range: tuple[int, int]) -> float:
    """Mirrors agent/scoring.py:score_completeness()'s raw signal, before
    _normalize() maps it onto the 1-5 band."""
    word_count = len(draft.split())
    low, high = target_word_range
    mid = (low + high) / 2
    half_width = (high - low) / 2
    word_fit = max(0.0, 1.0 - abs(word_count - mid) / (half_width * 2))

    header_count = len(re.findall(r"^#{1,3}\s+\S|^[A-Z][\w\s]{3,60}:?\s*$", draft, re.MULTILINE))
    header_fit = min(header_count / 3, 1.0)

    return word_fit * 0.6 + header_fit * 0.4


async def build_draft(topic: str) -> tuple[str, str]:
    ideas = generate_ideas(topic)
    idea = ideas[0]
    kb_chunks = retrieve(idea)
    time.sleep(4)  # ~15 RPM free-tier limit for Gemini Flash, between ideas call and draft call
    draft = await generate_draft(idea, kb_chunks)
    return idea, draft


async def build_samples() -> list[dict]:
    samples = []
    for i, topic in enumerate(TOPICS):
        if i > 0:
            time.sleep(4)  # ~15 RPM free-tier limit for Gemini Flash, between topics
        print(f"[{i+1}/{len(TOPICS)}] {topic} ...", end=" ", flush=True)
        idea, draft = await build_draft(topic)
        raw = _raw_completeness(draft, TARGET_WORD_RANGE)
        print(f"words={len(draft.split())} raw={raw:.4f}")
        samples.append({
            "topic": topic,
            "idea": idea,
            "word_count": len(draft.split()),
            "raw_completeness": round(raw, 4),
        })
    return samples


def print_table(samples: list[dict]) -> None:
    print("\nTopic | Words | Raw Completeness")
    print("------|-------|------------------")
    for s in samples:
        print(f"{s['topic'][:40]:40s} | {s['word_count']:5d} | {s['raw_completeness']:.4f}")


def recommend_bounds(samples: list[dict]) -> tuple[float, float]:
    raws = np.array([s["raw_completeness"] for s in samples])
    low = float(np.percentile(raws, LOW_PERCENTILE))
    high = float(np.percentile(raws, HIGH_PERCENTILE))
    return low, high


def save_result(samples: list[dict], low: float, high: float) -> None:
    result = {
        "recommended_completeness_low": round(low, 4),
        "recommended_completeness_high": round(high, 4),
        "low_percentile": LOW_PERCENTILE,
        "high_percentile": HIGH_PERCENTILE,
        "target_word_range": list(TARGET_WORD_RANGE),
        "model": require_env("GEMINI_MODEL"),
        "samples": samples,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    out_path = Path(__file__).parent / "calibration_result_completeness.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nResult saved to {out_path}")


def main() -> None:
    print("=== Completeness (Word-Fit + Header-Fit) Calibration ===")
    samples = asyncio.run(build_samples())

    print_table(samples)

    low, high = recommend_bounds(samples)
    print(f"\nRecommended COMPLETENESS_LOW  = {low:.4f}  (p{LOW_PERCENTILE})")
    print(f"Recommended COMPLETENESS_HIGH = {high:.4f}  (p{HIGH_PERCENTILE})")
    print("\nUpdate COMPLETENESS_LOW / COMPLETENESS_HIGH in config/constants.py")

    save_result(samples, low, high)


if __name__ == "__main__":
    main()
