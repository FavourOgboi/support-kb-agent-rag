import os
from typing import List

from langchain_community.vectorstores import FAISS
from scripts.embed_store_faiss import store_in_faiss, load_faiss

try:
	# Prefer the dedicated langchain-cohere integration
	from langchain_cohere import CohereEmbeddings
except ImportError as e:
	raise ImportError(
		"langchain-cohere is required for Cohere embeddings. "
		"Install it with 'pip install -U langchain-cohere'."
	) from e

try:
	from langchain_core.documents import Document
except ImportError:
	from langchain.docstore.document import Document

def get_cohere_embeddings():
    api_key = os.environ.get("COHERE_API_KEY")
    if not api_key:
        raise ValueError("Set your Cohere API key in the COHERE_API_KEY environment variable.")
    return CohereEmbeddings(model="embed-english-v3.0", cohere_api_key=api_key)

def store_in_faiss_wrapper(
    docs: List["Document"],
    persist_path: str = "faiss_index"
):
    # Use the FAISS wrapper for storing embeddings
    return store_in_faiss(docs, persist_path=persist_path)

if __name__ == "__main__":
    import argparse
    import pickle

    parser = argparse.ArgumentParser(description="Embed and store document chunks in FAISS")
    parser.add_argument("--chunks_file", type=str, required=True, help="Pickle file with document chunks")
    parser.add_argument("--persist_path", type=str, default="faiss_index")
    args = parser.parse_args()

    with open(args.chunks_file, "rb") as f:
        chunks = pickle.load(f)

    vectordb = store_in_faiss_wrapper(chunks, persist_path=args.persist_path)
    print(f"Stored {len(chunks)} chunks in FAISS at {args.persist_path}")