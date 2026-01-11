"""
Debug script to test the embedding pipeline outside of Jupyter.
Run from the support_kb_agent_demo directory:
    python scripts/debug_embed.py
"""
import os
import sys
import time
import traceback

# Add the parent directory to sys.path so we can import from scripts
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.ingest import load_pdf, chunk_documents
from scripts.embed_store import store_in_chroma

# Paths relative to support_kb_agent_demo directory
PDF_PATH = "data/winter-sports.pdf"
PERSIST_DIR = "chroma_db"


def main() -> None:
    print(f"Loading PDF: {PDF_PATH}")
    start = time.time()
    docs = load_pdf(PDF_PATH)
    print(f"Loaded {len(docs)} docs in {time.time() - start:.2f}s")

    print("Chunking documents...")
    start = time.time()
    chunks = chunk_documents(docs)
    print(f"Produced {len(chunks)} chunks in {time.time() - start:.2f}s")

    print("Embedding & storing in Chroma...")
    start = time.time()
    try:
        vectordb = store_in_chroma(chunks, persist_directory=PERSIST_DIR)
        print(f"Embedding+store completed in {time.time() - start:.2f}s")
        print(f"Vector DB: {vectordb}")
    except Exception:
        print("ERROR during embedding/store:")
        traceback.print_exc()


if __name__ == "__main__":
    main()

