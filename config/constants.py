import os
from pathlib import Path

_ROOT = Path(__file__).parent.parent

# Knowledge base / vector store
KB_PATH = _ROOT / "kb" / "software-testing-guide-book.pdf"
CHROMA_PATH = _ROOT / "chroma_db"
COLLECTION_NAME = "knowledge_base"

# Retrieval
TOP_K = 5
KB_SIMILARITY_THRESHOLD = float(os.getenv("RAG_SIMILARITY_THRESHOLD", "0.5"))  # run scripts/calibrate_threshold.py to tune

# Improvement loop
MAX_ITERATIONS = 2
SCORE_THRESHOLD = 3.5

# Evaluation
CRITERIA = ["Clarity", "Relevance", "Completeness",
            "Accuracy", "Actionability", "Retrieval Relevance"]

# Generator's target word count for the draft (agent/generator.py PROMPT_TEMPLATE
TARGET_WORD_RANGE = (600, 900)

# Deterministic scoring bounds (agent/scoring.py) — map a raw cosine similarity
RETRIEVAL_RELEVANCE_LOW = 0.5998
RETRIEVAL_RELEVANCE_HIGH = 0.7158

# Calibrated via scripts/calibrate_clarity.py
CLARITY_LOW = 26.47
CLARITY_HIGH = 39.47

# Calibrated via scripts/calibrate_relevance.py
RELEVANCE_LOW = 0.8144
RELEVANCE_HIGH = 0.8449

# Calibrated via scripts/calibrate_completeness.py
COMPLETENESS_LOW = 0.9214
COMPLETENESS_HIGH = 0.9932

# Criteria judged by the LLM (agent/evaluator.py) rather than deterministic
LLM_JUDGED_CRITERIA = {"Accuracy", "Actionability"}

# Ingest chunking / batching (free-tier rate limit)
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
BATCH_SIZE = 90       # stay under 100 req/min free tier limit
BATCH_DELAY = 65      # seconds to wait between batches

FIRECRAWL_CONCURRENCY = 2   # semaphore + thread pool cap
FIRECRAWL_LIMIT = 2         # results per search query (= credits per query)
FETCH_MAX_CHARS = 3000
