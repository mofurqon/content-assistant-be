# /ingest

Run the KB ingestion pipeline.

## What it does
Reads the PDF from `kb/software-testing-guide-book.pdf`, chunks it, embeds it using Gemini,
and stores the vectors in `chroma_db/`.

## When to use
- First time setup
- When the KB PDF has been replaced or updated

## Command
```bash
python -m ingest.ingest
```
Run from the project root so the `config` / `core` packages resolve.

## Expected output
```
Loading PDF...
Splitting into chunks...
Total chunks: XX
Embedding and storing in Chroma...
Done. KB is ready.
```

## Notes
- Chroma DB is persistent — re-running will overwrite existing vectors
- Make sure GEMINI_API_KEY is set in .env before running
