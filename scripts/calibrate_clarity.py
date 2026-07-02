"""
One-time offline script to find realistic Flesch Reading Ease bounds for
score_clarity() (agent/scoring.py), replacing the guessed CLARITY_LOW/HIGH
placeholders in constants.py.

Generates a batch of real drafts through the actual pipeline steps (ideas ->
retrieve -> generate_draft), scores each with the same textstat call
score_clarity() uses, and recommends bounds from the observed distribution.

Usage:
    python scripts/calibrate_clarity.py

After running, set CLARITY_LOW / CLARITY_HIGH in config/constants.py to the
recommended values printed below.
"""

import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import textstat

# Allow importing from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.generator import generate_draft
from agent.ideas import generate_ideas
from agent.retriever import retrieve
from config.settings import require_env

# Mostly KB-domain topics (what the pipeline is actually tuned for) plus a
# couple of off-KB topics, so the calibration reflects the real spread of
# input rather than one narrow subject.
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


async def build_draft(topic: str) -> tuple[str, int, float]:
    ideas = generate_ideas(topic)
    idea = ideas[0]
    kb_chunks = retrieve(idea)
    time.sleep(4)  # ~15 RPM free-tier limit for Gemini Flash (same pacing as calibrate_threshold.py) between ideas call and draft call
    draft = await generate_draft(idea, kb_chunks)
    fre = textstat.flesch_reading_ease(draft)
    return idea, len(draft.split()), fre


async def build_samples() -> list[dict]:
    samples = []
    for i, topic in enumerate(TOPICS):
        if i > 0:
            time.sleep(4)  # ~15 RPM free-tier limit for Gemini Flash, between topics
        print(f"[{i+1}/{len(TOPICS)}] {topic} ...", end=" ", flush=True)
        idea, word_count, fre = await build_draft(topic)
        print(f"words={word_count} FRE={fre:.1f}")
        samples.append({
            "topic": topic,
            "idea": idea,
            "word_count": word_count,
            "fre": round(fre, 2),
        })
    return samples


def print_table(samples: list[dict]) -> None:
    print("\nTopic | Words | FRE")
    print("------|-------|----")
    for s in samples:
        print(f"{s['topic'][:40]:40s} | {s['word_count']:5d} | {s['fre']:.1f}")


def recommend_bounds(samples: list[dict]) -> tuple[float, float]:
    scores = np.array([s["fre"] for s in samples])
    low = float(np.percentile(scores, LOW_PERCENTILE))
    high = float(np.percentile(scores, HIGH_PERCENTILE))
    return low, high


def save_result(samples: list[dict], low: float, high: float) -> None:
    result = {
        "recommended_clarity_low": round(low, 2),
        "recommended_clarity_high": round(high, 2),
        "low_percentile": LOW_PERCENTILE,
        "high_percentile": HIGH_PERCENTILE,
        "model": require_env("GEMINI_MODEL"),
        "samples": samples,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    out_path = Path(__file__).parent / "calibration_result_clarity.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nResult saved to {out_path}")


def main() -> None:
    print("=== Clarity (Flesch Reading Ease) Calibration ===")
    samples = asyncio.run(build_samples())

    print_table(samples)

    low, high = recommend_bounds(samples)
    print(f"\nRecommended CLARITY_LOW  = {low:.2f}  (p{LOW_PERCENTILE})")
    print(f"Recommended CLARITY_HIGH = {high:.2f}  (p{HIGH_PERCENTILE})")
    print("\nUpdate CLARITY_LOW / CLARITY_HIGH in config/constants.py")

    save_result(samples, low, high)


if __name__ == "__main__":
    main()
