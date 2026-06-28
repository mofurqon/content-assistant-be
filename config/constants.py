from pathlib import Path

_ROOT = Path(__file__).parent.parent

# Knowledge base / vector store
KB_PATH = _ROOT / "kb" / "software-testing-guide-book.pdf"
CHROMA_PATH = _ROOT / "chroma_db"
COLLECTION_NAME = "knowledge_base"

# Retrieval
TOP_K = 5
KB_SIMILARITY_THRESHOLD = 0.5  # minimum cosine similarity to include a chunk (Gemini embedding-001 scores lower than OpenAI)

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
FIRECRAWL_LIMIT = 2
FETCH_MAX_CHARS = 3000
