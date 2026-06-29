#!/bin/bash
set -e

CHROMA_DIR="${CHROMA_DB_PATH:-/app/chroma_db}"

echo "Checking Chroma at $CHROMA_DIR ..."
if python -c "
import sys, chromadb
try:
    client = chromadb.PersistentClient(path='$CHROMA_DIR')
    col = client.get_collection('knowledge_base')
    count = col.count()
    print(f'  Chroma OK: {count} chunks found')
    sys.exit(0 if count > 0 else 1)
except Exception as e:
    print(f'  Chroma not ready: {e}')
    sys.exit(1)
"; then
    echo "Skipping ingest - Chroma already populated."
else
    echo "Running PDF ingest (first boot or empty volume)..."
    python -m ingest.ingest
fi

exec uvicorn api.main:app --host 0.0.0.0 --port "${PORT:-8000}"
