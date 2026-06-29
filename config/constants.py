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
SCORE_THRESHOLD = 4.0

# Evaluation
CRITERIA = ["Clarity", "Relevance", "Completeness",
            "Accuracy", "Actionability", "KB Alignment"]

# Ingest chunking / batching (free-tier rate limit)
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
BATCH_SIZE = 90       # stay under 100 req/min free tier limit
BATCH_DELAY = 65      # seconds to wait between batches

# Web research
# Free tier: 1000 credits/month, 2 concurrent requests.
# Budget: 3 queries × FIRECRAWL_LIMIT=2 results = 6 credits per research() call → ~166 calls/month.
FIRECRAWL_CONCURRENCY = 2   # semaphore + thread pool cap
FIRECRAWL_LIMIT = 2         # results per search query (= credits per query)
FETCH_MAX_CHARS = 3000
