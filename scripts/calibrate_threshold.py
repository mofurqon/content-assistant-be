"""
One-time offline script to find the optimal cosine similarity threshold
for Gemini gemini-embedding-2, replacing the hardcoded value in constants.py.

Usage:
    python scripts/calibrate_threshold.py

After running, set RAG_SIMILARITY_THRESHOLD in your .env to the recommended value.
"""

import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import chromadb
import numpy as np
from sklearn.metrics import precision_recall_curve

# Allow importing from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.constants import CHROMA_PATH, COLLECTION_NAME
from config.settings import require_env
from core.llm import get_embeddings, get_llm

MAX_CHUNKS = 80
QUESTIONS_PER_CHUNK = 3
NEGATIVES_PER_QUESTION = 3

QA_PROMPT = (
    "Generate exactly 3 short questions that are directly answered by the text below.\n"
    "Return only the questions, one per line, no numbering, no extra text.\n\n"
    "Text: {chunk}"
)


def load_chunks() -> tuple[list[str], list[list[float]]]:
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    col = client.get_collection(COLLECTION_NAME)
    data = col.get(include=["embeddings", "documents"])
    return data["documents"], data["embeddings"]


def generate_questions(chunk: str, llm) -> list[str]:
    response = llm.invoke(QA_PROMPT.format(chunk=chunk))
    lines = [l.strip() for l in response.content.splitlines() if l.strip()]
    return lines[:QUESTIONS_PER_CHUNK]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def embed_questions(questions: list[str]) -> dict[str, list[float]]:
    embeddings_client = get_embeddings()
    cache: dict[str, list[float]] = {}
    for i, q in enumerate(questions):
        if q not in cache:
            if i > 0:
                time.sleep(0.5)
            cache[q] = embeddings_client.embed_query(q)
    return cache


def build_pairs(
    chunks: list[str],
    chunk_embeddings: list[list[float]],
    llm,
) -> tuple[list[float], list[int]]:
    scores: list[float] = []
    labels: list[int] = []
    all_questions: list[str] = []
    question_to_source_idx: list[int] = []

    print(f"\nGenerating questions for {len(chunks)} chunks...")
    chunk_questions: list[list[str]] = []
    for i, chunk in enumerate(chunks):
        if i > 0:
            time.sleep(4)  # ~15 RPM free-tier limit for Gemini Flash
        qs = generate_questions(chunk, llm)
        chunk_questions.append(qs)
        all_questions.extend(qs)
        question_to_source_idx.extend([i] * len(qs))
        print(f"  [{i+1}/{len(chunks)}] {len(qs)} questions generated", end="\r")

    print(f"\nEmbedding {len(all_questions)} unique questions...")
    q_embedding_cache = embed_questions(list(dict.fromkeys(all_questions)))

    print("Computing cosine similarities...")
    other_indices = list(range(len(chunks)))
    for q_idx, (question, source_idx) in enumerate(zip(all_questions, question_to_source_idx)):
        q_emb = q_embedding_cache[question]

        # Positive pair
        scores.append(cosine_similarity(q_emb, chunk_embeddings[source_idx]))
        labels.append(1)

        # Negative pairs
        neg_pool = [i for i in other_indices if i != source_idx]
        for neg_idx in random.sample(neg_pool, min(NEGATIVES_PER_QUESTION, len(neg_pool))):
            scores.append(cosine_similarity(q_emb, chunk_embeddings[neg_idx]))
            labels.append(0)

    return scores, labels


def find_best_threshold(
    scores: list[float], labels: list[int]
) -> tuple[float, float, float, float]:
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    best_idx = int(np.argmax(f1[:-1]))  # last element has no matching threshold
    return float(thresholds[best_idx]), float(precision[best_idx]), float(recall[best_idx]), float(f1[best_idx])


def print_table(scores: list[float], labels: list[int], best_threshold: float) -> None:
    _, _, thresholds = precision_recall_curve(labels, scores)
    # Sample ~10 evenly spaced thresholds for the table
    step = max(1, len(thresholds) // 10)
    sampled = sorted(set(list(thresholds[::step]) + [best_threshold]))

    print("\nThreshold | Precision | Recall | F1")
    print("----------|-----------|--------|----")
    for t in sampled:
        preds = [1 if s >= t else 0 for s in scores]
        tp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 1)
        fp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 0)
        fn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 1)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        marker = "  ← best F1" if abs(t - best_threshold) < 1e-6 else ""
        print(f"{t:.2f}      | {prec:.2f}      | {rec:.2f}   | {f1:.2f}{marker}")


def save_result(
    threshold: float,
    f1: float,
    precision: float,
    recall: float,
    pairs_evaluated: int,
) -> None:
    result = {
        "recommended_threshold": round(threshold, 4),
        "f1": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "model": require_env("GEMINI_EMBED_MODEL"),
        "pairs_evaluated": pairs_evaluated,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    out_path = Path(__file__).parent / "calibration_result.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nResult saved to {out_path}")


def main() -> None:
    print("=== Threshold Calibration ===")
    print(f"Loading chunks from Chroma ({CHROMA_PATH})...")

    documents, embeddings = load_chunks()
    total = len(documents)
    print(f"Found {total} chunks.")

    if total > MAX_CHUNKS:
        indices = random.sample(range(total), MAX_CHUNKS)
        documents = [documents[i] for i in indices]
        embeddings = [embeddings[i] for i in indices]
        print(f"Sampled {MAX_CHUNKS} chunks (out of {total}).")

    llm = get_llm(temperature=0.3)
    scores, labels = build_pairs(documents, embeddings, llm)

    best_threshold, precision, recall, f1 = find_best_threshold(scores, labels)
    print_table(scores, labels, best_threshold)

    print(f"\n✅ Recommended threshold: {best_threshold:.2f}")
    print(f"   Precision: {precision:.2f} | Recall: {recall:.2f} | F1: {f1:.2f}")
    print(f"\n   Update RAG_SIMILARITY_THRESHOLD in your .env")

    save_result(best_threshold, f1, precision, recall, len(scores))


if __name__ == "__main__":
    main()
