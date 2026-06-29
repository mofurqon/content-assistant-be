# AI Content Assistant

An AI agent that generates high-quality articles from a topic using a RAG pipeline, self-evaluation loop, and web research enrichment.

Built as a case study prototype with Google Gemini, LangChain, ChromaDB, Streamlit, and FastAPI.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Interface Layer                      │
│   ui/app.py (Streamlit)     api/main.py (FastAPI)       │
└───────────────────┬─────────────────────────────────────┘
                    │ both call
┌───────────────────▼─────────────────────────────────────┐
│              services/content_pipeline.py               │
│           ContentPipelineService (use cases)            │
└───────────────────┬─────────────────────────────────────┘
                    │ wraps
┌───────────────────▼─────────────────────────────────────┐
│                     agent/  (step logic)                 │
│  ideas → retriever → generator → evaluator →            │
│  improver → researcher → finalizer                      │
└───────────────────┬─────────────────────────────────────┘
                    │ uses
┌───────────────────▼─────────────────────────────────────┐
│   core/llm.py (Gemini clients)   config/ (constants)    │
│   ChromaDB (vectors)             Firecrawl (web search)  │
└─────────────────────────────────────────────────────────┘
```

**Layer dependency rule:** `api/` and `ui/` call `services/` only. `agent/` never imports from layers above it. `domain/models.py` is imported by all layers.

---

## Pipeline Steps

| Step | Module | What it does |
|------|--------|-------------|
| 1 | `agent/ideas.py` | LLM generates 5 article ideas from user topic |
| 2 | `agent/retriever.py` | Embeds selected idea, fetches top-5 KB chunks from Chroma |
| 3 | `agent/generator.py` | Drafts a 600–900 word article using KB chunks as context |
| 4 | `agent/evaluator.py` | Scores draft on 6 criteria (Clarity, Relevance, Completeness, Accuracy, Actionability, KB Alignment) |
| 5 | `agent/improver.py` | Rewrites weak drafts; loops up to 2× or until avg score ≥ 4.0 |
| 6 | `agent/researcher.py` | Generates 3 web queries, fetches & summarizes external sources |
| 7 | `agent/finalizer.py` | Merges improved draft + research into final 700–1000 word article + image prompt |

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env`:

```
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-2.0-flash
GEMINI_EMBED_MODEL=embedding-001
FIRECRAWL_API_KEY=your_key_here   # optional — mocked if omitted
```

### 3. Ingest the knowledge base

Run once to chunk and embed the PDF into ChromaDB:

```bash
python -m ingest.ingest
```

Expected output:
```
Loading PDF: kb/software-testing-guide-book.pdf
  Loaded 42 pages
  Split into 187 chunks
Embedding and storing in Chroma (batched for free-tier rate limit)...
  Batch 1/3 (90 chunks)...
  Rate limit pause (65s)...
  Batch 2/3 (90 chunks)...
  Rate limit pause (65s)...
  Batch 3/3 (7 chunks)...
  Stored 187 chunks in chroma_db
Ingest complete.
```

---

## Running

### Streamlit UI

```bash
streamlit run ui/app.py
```

Opens at `http://localhost:8501`. Full interactive pipeline with streaming output.

### FastAPI

```bash
uvicorn api.main:app --reload --port 8000
```

Interactive docs at `http://localhost:8000/docs`.

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/v1/ideas` | Generate 5 article ideas from a topic |
| POST | `/v1/draft` | Stream draft article tokens (SSE) |
| POST | `/v1/finalize` | Stream final article tokens (SSE) |
| POST | `/v1/pipeline` | Run full pipeline, return JSON result |

### Examples

**Generate ideas**
```bash
curl -X POST http://localhost:8000/v1/ideas \
  -H "Content-Type: application/json" \
  -d '{"topic": "software testing best practices"}'
```
```json
{
  "ideas": [
    "Why Shift-Left Testing Reduces Bug Costs by 10x",
    "The Art of Writing Testable Code from Day One",
    "Unit vs Integration vs E2E: Choosing the Right Test for the Job",
    "Test-Driven Development in Practice: A Step-by-Step Guide",
    "How to Build a Reliable CI Pipeline That Developers Actually Trust"
  ]
}
```

**Run full pipeline**
```bash
curl -X POST http://localhost:8000/v1/pipeline \
  -H "Content-Type: application/json" \
  -d '{"idea": "Why Shift-Left Testing Reduces Bug Costs by 10x"}'
```
Returns a `PipelineResult` JSON with `kb_chunks`, `draft`, `evaluations`, `research`, `article`, and `image_prompt`.

**Stream draft (SSE)**
```bash
curl -X POST http://localhost:8000/v1/draft \
  -H "Content-Type: application/json" \
  -d '{"idea": "Why Shift-Left Testing Reduces Bug Costs by 10x"}'
```
Response is a stream of `data: <token>` lines (Server-Sent Events).

---

## Sample Input / Output

**Input topic:** `software testing`

**Generated ideas (Step 1):**
1. Why Shift-Left Testing Reduces Bug Costs by 10x
2. The Art of Writing Testable Code from Day One
3. Unit vs Integration vs E2E: Choosing the Right Test for the Job
4. Test-Driven Development in Practice: A Step-by-Step Guide
5. How to Build a Reliable CI Pipeline That Developers Actually Trust

**Selected idea:** *Why Shift-Left Testing Reduces Bug Costs by 10x*

**KB retrieval (Step 2):** 5 chunks retrieved from the software testing guide (relevance scores ~0.78–0.85)

**Draft evaluation (Step 4):**
| Criterion | Score |
|-----------|-------|
| Clarity | 4/5 |
| Relevance | 5/5 |
| Completeness | 3/5 |
| Accuracy | 4/5 |
| Actionability | 3/5 |
| KB Alignment | 4/5 |
| **Average** | **3.83/5** |

*Below threshold (4.0) → one improvement iteration runs.*

**Post-improvement evaluation:**
| Criterion | Score |
|-----------|-------|
| Clarity | 4/5 |
| Relevance | 5/5 |
| Completeness | 4/5 |
| Accuracy | 5/5 |
| Actionability | 4/5 |
| KB Alignment | 5/5 |
| **Average** | **4.5/5** ✅ |

**Web research (Step 6):** 3 queries generated, key findings summarized in 5 bullet points

**Final article (Step 7):** 850-word structured article combining improved draft + research insights

**Image prompt (Step 7):**
```
A split-screen illustration showing a developer catching a small bug early in the 
development cycle on the left, versus a team of engineers firefighting a production 
outage caused by the same bug on the right. The left panel is calm, green-tinted, 
with clean code on a monitor. The right panel is chaotic, red-alert atmosphere, with 
dashboards showing errors. Style: flat design, professional, tech-themed.
```

---

## Trade-offs

### LLM provider: single Gemini key for both LLM and embeddings
**Pro:** One API key, one billing account, consistent latency profile.  
**Con:** Couples text generation and embedding to the same provider — if Gemini's embedding quality degrades or pricing changes, both concerns are affected together. A production system might separate these (e.g., OpenAI embeddings + Anthropic generation).

### Local ChromaDB (file-based persistence)
**Pro:** Zero infrastructure — just a directory. No network hop for vector search. Fast iteration.  
**Con:** Not horizontally scalable. A single pod owns the `chroma_db/` directory; deploying two instances means two divergent vector stores. Production would swap in a managed vector DB (Pinecone, Weaviate, Qdrant).

### Blocking improvement loop (max 2 iterations)
**Pro:** Bounded latency — worst case is 2× eval + 2× generation LLM calls.  
**Con:** The stopping threshold (avg ≥ 4.0) and iteration cap (2) are fixed constants, not learned from user feedback. A low-quality topic that consistently scores 3.9 will always exhaust the loop without improvement.

### Firecrawl web research (optional / mocked)
**Pro:** Graceful degradation — no crash if key is absent, just mock data.  
**Con:** Mock results mean the "research" section of the final article adds no real value. The pipeline doesn't signal to the user that research was skipped.

### Streamlit and FastAPI share the service layer (same process model)
**Pro:** No network round-trip between UI and backend. Streaming generators work natively in both.  
**Con:** You can't scale the API independently from the UI. A high-traffic API server and a single-user Streamlit session would compete for the same process resources.

### Synchronous FastAPI endpoints
**Pro:** Simple to reason about; no async complexity.  
**Con:** Each LLM call blocks a thread for several seconds. Under concurrent load, the thread pool exhausts quickly. Production would use `async def` + `httpx.AsyncClient` or offload to a task queue (Celery, ARQ).

---

## Project Structure

```
content-assistant/
├── domain/models.py           # Pydantic domain entities
├── services/content_pipeline.py  # Application use cases
├── api/                       # FastAPI HTTP interface
├── ui/app.py                  # Streamlit frontend
├── agent/                     # Core pipeline step implementations
├── config/                    # Env + tuning constants
├── core/llm.py                # Cached LLM/embedding factory
├── ingest/ingest.py           # One-time KB ingestion
└── kb/                        # Source PDF
```
