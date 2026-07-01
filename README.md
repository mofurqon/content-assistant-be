# AI Content Assistant

An AI agent that generates high-quality articles from a topic using a RAG pipeline, self-evaluation loop, and web research enrichment.

Built as a case study prototype with Google Gemini, LangGraph, LangChain, ChromaDB, and FastAPI.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Interface Layer                      │
│                  api/main.py (FastAPI)                   │
└───────────────────┬─────────────────────────────────────┘
                    │ drives
┌───────────────────▼─────────────────────────────────────┐
│                  agent/graph.py (LangGraph)              │
│           Stateful pipeline graph with checkpointing     │
└───────────────────┬─────────────────────────────────────┘
                    │ step nodes
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

**Layer dependency rule:** `api/` drives `agent/graph.py` directly. `agent/` never imports from layers above it. `domain/models.py` is imported by all layers.

---

## Pipeline Steps

| Step | Module | What it does |
|------|--------|-------------|
| 1 | `agent/ideas.py` | LLM generates 5 article ideas from user topic |
| 2 | `agent/retriever.py` | Embeds selected idea, fetches top-5 KB chunks from Chroma |
| 3 | `agent/generator.py` | Drafts a 600–900 word article using KB chunks as context |
| 4 | `agent/evaluator.py` | Scores draft on 6 criteria (Clarity, Relevance, Completeness, Accuracy, Actionability, Retrieval Relevance) |
| 5 | `agent/improver.py` | Rewrites weak drafts; loops up to 2× or until avg score ≥ 4.0 |
| 6 | `agent/researcher.py` | Generates 3 web queries, fetches & summarizes external sources |
| 7 | `agent/finalizer.py` | Merges improved draft + research into final 700–1000 word article + image prompt |

The pipeline pauses after step 3 (draft generation) to collect optional human feedback before continuing to research and finalization.

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

```env
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-2.0-flash
GEMINI_EMBED_MODEL=gemini-embedding-2
FIRECRAWL_API_KEY=your_key_here   # optional — mocked if omitted
LOG_LEVEL=INFO
RAG_SIMILARITY_THRESHOLD=0.5
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

```bash
uvicorn api.main:app --reload --port 8000
```

Interactive docs at `http://localhost:8000/docs`.

---

## API Reference

The API is session-based. Each pipeline run is tied to a `session_id` returned on session creation.

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/v1/ideas` | Create a session and stage a topic |
| GET | `/v1/ideas/{session_id}/stream` | Stream idea generation (SSE) |
| POST | `/v1/pipeline/{session_id}` | Stage a selected idea + content criteria |
| GET | `/v1/pipeline/{session_id}/stream` | Stream draft then final article (SSE) |
| POST | `/v1/pipeline/{session_id}/resume` | Stage human feedback, then re-call stream |

### SSE event types

| `type` | When emitted | Payload |
|--------|-------------|---------|
| `token` | During LLM generation | `{ "content": "..." }` |
| `node` | When a pipeline node starts | `{ "name": "generate" \| "evaluate" \| ... }` |
| `interrupted` | After ideas are ready | `{ "ideas": ["...", ...] }` |
| `awaiting_feedback` | After draft is ready | `{ "draft": "..." }` |
| `done` | Pipeline complete | `{ "result": { ... } }` |

### Examples

**Step 1 — Create a session**
```bash
curl -X POST http://localhost:8000/v1/ideas \
  -H "Content-Type: application/json" \
  -d '{"topic": "software testing best practices"}'
```
```json
{ "session_id": "a1b2c3d4-..." }
```

**Step 2 — Stream idea generation**
```bash
curl http://localhost:8000/v1/ideas/a1b2c3d4-.../stream
```
Streams `token` events, then emits `interrupted` with the ideas list.

**Step 3 — Stage selected idea**
```bash
curl -X POST http://localhost:8000/v1/pipeline/a1b2c3d4-... \
  -H "Content-Type: application/json" \
  -d '{
    "selected_idea": "Why Shift-Left Testing Reduces Bug Costs by 10x",
    "criteria": {
      "target_audience": "software engineers",
      "content_type": "Article",
      "tone": "Professional and informative"
    }
  }'
```

**Step 4 — Stream draft**
```bash
curl http://localhost:8000/v1/pipeline/a1b2c3d4-.../stream
```
Streams `token` and `node` events, then emits `awaiting_feedback` with the draft.

**Step 5 — Submit feedback (optional)**
```bash
curl -X POST http://localhost:8000/v1/pipeline/a1b2c3d4-.../resume \
  -H "Content-Type: application/json" \
  -d '{"human_feedback": "Add more examples and make it more concise."}'
```

**Step 6 — Stream final article**
```bash
curl http://localhost:8000/v1/pipeline/a1b2c3d4-.../stream
```
Streams tokens for the final article, then emits `done` with the full result.

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
| Retrieval Relevance | 4/5 |
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
| Retrieval Relevance | 4/5 |
| **Average** | **4.33/5** ✅ |

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

### LangGraph stateful pipeline with checkpointing
**Pro:** Human-in-the-loop is first-class — the graph pauses at natural breakpoints (after ideas, after draft) and resumes with user input. State is persisted per session via the in-memory checkpointer.  
**Con:** The in-memory checkpointer does not survive server restarts. A Railway redeploy clears all in-flight sessions. Production would use a persistent checkpointer (Postgres, Redis).

### Blocking improvement loop (max 2 iterations)
**Pro:** Bounded latency — worst case is 2× eval + 2× generation LLM calls.  
**Con:** The stopping threshold (avg ≥ 4.0) and iteration cap (2) are fixed constants, not learned from user feedback. A low-quality topic that consistently scores 3.9 will always exhaust the loop without improvement.

### Firecrawl web research (optional / mocked)
**Pro:** Graceful degradation — no crash if key is absent, just mock data.  
**Con:** Mock results mean the "research" section of the final article adds no real value. The pipeline doesn't signal to the user that research was skipped.

### Async FastAPI with SSE streaming
**Pro:** Non-blocking — LLM tokens stream to the client without holding a thread. Multiple concurrent sessions are handled efficiently.  
**Con:** SSE is one-directional and stateless; clients that disconnect mid-stream lose progress and must re-call the stream endpoint (which re-runs from the last checkpoint).

---

## Project Structure

```
content-assistant/
├── domain/models.py              # Pydantic domain entities
├── api/                          # FastAPI HTTP interface
│   ├── main.py                   # App + CORS + rate limit setup
│   ├── middleware/rate_limit.py  # slowapi wrapper (Railway only)
│   ├── routes/                   # health, ideas, content
│   └── schemas/                  # Request / response models
├── agent/                        # Pipeline step implementations
│   ├── graph.py                  # LangGraph pipeline definition
│   ├── ideas.py
│   ├── retriever.py
│   ├── generator.py
│   ├── evaluator.py
│   ├── improver.py
│   ├── researcher.py
│   └── finalizer.py
├── config/
│   ├── settings.py               # Env loading + require_env
│   └── constants.py              # Tuning knobs (TOP_K, thresholds, etc.)
├── core/llm.py                   # Cached LLM/embedding factory
├── ingest/ingest.py              # One-time KB ingestion
├── scripts/
│   ├── start.sh                  # Docker entrypoint
│   └── calibrate_threshold.py   # RAG threshold tuning
├── kb/                           # Source PDF
├── Dockerfile
└── railway.toml
```
