import time
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from core.llm import get_embeddings
from config.constants import (
    KB_PATH,
    CHROMA_PATH,
    COLLECTION_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    BATCH_SIZE,
    BATCH_DELAY,
)


def ingest():
    print(f"Loading PDF: {KB_PATH}")
    loader = PyPDFLoader(str(KB_PATH))
    docs = loader.load()
    print(f"  Loaded {len(docs)} pages")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    print(f"  Split into {len(chunks)} chunks")

    embeddings = get_embeddings()

    print("Embedding and storing in Chroma (batched for free-tier rate limit)...")
    vectorstore = None
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"  Batch {batch_num}/{total_batches} ({len(batch)} chunks)...")

        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                collection_name=COLLECTION_NAME,
                persist_directory=str(CHROMA_PATH),
                # Without this, Chroma defaults to L2 distance. Its relevance-score
                # formula for L2 (1 - distance/sqrt(2)) is NOT cosine similarity and
                # reads systematically lower — retriever.py and calibrate_threshold.py
                # both assume true cosine similarity, so the collection must be built
                # with this metric or every downstream score/threshold is miscalibrated.
                collection_metadata={"hnsw:space": "cosine"},
            )
        else:
            vectorstore.add_documents(batch)

        if i + BATCH_SIZE < len(chunks):
            print(f"  Rate limit pause ({BATCH_DELAY}s)...")
            time.sleep(BATCH_DELAY)

    print(f"  Stored {vectorstore._collection.count()} chunks in {CHROMA_PATH}")
    print("Ingest complete.")


if __name__ == "__main__":
    ingest()
