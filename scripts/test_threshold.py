"""
Interactive script to observe how many KB chunks pass at different
similarity thresholds for real user-style queries.

Usage:
    python scripts/test_threshold.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_chroma import Chroma
from config.constants import CHROMA_PATH, COLLECTION_NAME
from core.llm import get_embeddings

queries = [
    "What is regression testing?",
    "How do you write a test case?",
    "What is the difference between unit and integration testing?",
    # add queries that reflect your actual users
]

embeddings_client = get_embeddings()
vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings_client,
    persist_directory=str(CHROMA_PATH),
)

for threshold in [0.60, 0.62, 0.65, 0.66]:
    print(f"\n--- Threshold: {threshold} ---")
    for q in queries:
        scored = vectorstore.similarity_search_with_relevance_scores(q, k=5)
        passing = [score for _, score in scored if score >= threshold]
        all_scores = [round(score, 4) for _, score in scored]
        print(f"  '{q[:40]}...' → {len(passing)}/5 passed | scores: {all_scores}")
