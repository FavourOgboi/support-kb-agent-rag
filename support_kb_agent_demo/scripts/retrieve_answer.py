import os
from langchain_community.vectorstores import FAISS
from scripts.embed_store_faiss import load_faiss

try:
    # Prefer the dedicated langchain-cohere integration
    from langchain_cohere import CohereEmbeddings
except ImportError as e:
    raise ImportError(
        "langchain-cohere is required for Cohere embeddings. "
        "Install it with 'pip install -U langchain-cohere'."
    ) from e

from langchain_openai import ChatOpenAI

def get_cohere_embeddings():
    api_key = os.environ.get("COHERE_API_KEY")
    if not api_key:
        raise ValueError("Set your Cohere API key in the COHERE_API_KEY environment variable.")
    return CohereEmbeddings(model="embed-english-v3.0", cohere_api_key=api_key)

def get_openai_llm():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Set your OpenAI API key in the OPENAI_API_KEY environment variable.")
    # I am doing this because we have switched to OpenAI for LLM responses to avoid Gemini API issues.
    return ChatOpenAI(openai_api_key=api_key, model="gpt-3.5-turbo", temperature=0)

def load_faiss_db(persist_path="faiss_index"):
    return load_faiss(persist_path)

def build_qa_chain(vectordb, llm):
	"""Return a simple callable that mimics RetrievalQA's interface.

	This avoids relying on langchain.chains / langchain_community.chains,
	which may not be available in newer LangChain versions.
	"""

	def _qa(query: dict):
	    # Support the existing calling convention: {"query": question}
	    question = query.get("query") if isinstance(query, dict) else str(query)

	    docs = vectordb.similarity_search(question, k=5)

	    context = "\n\n".join(doc.page_content for doc in docs)
	    prompt = (
	        "You are a helpful support agent. Use the provided context to answer the question. "
	        "Cite sources using [source] notation where appropriate.\n\n"
	        f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
	    )

	    response_msg = llm.invoke(prompt)

	    return {
	        "result": response_msg.content,
	        "source_documents": docs,
	    }

	return _qa

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Retrieve and answer questions from FAISS vector store")
    parser.add_argument("--persist_path", type=str, default="faiss_index")
    parser.add_argument("--question", type=str, required=True)
    args = parser.parse_args()

    vectordb = load_faiss_db(args.persist_path)
    llm = get_openai_llm()
    qa_chain = build_qa_chain(vectordb, llm)

    result = qa_chain({"query": args.question})
    print("Answer:\n", result["result"])
    print("\nSources:")
    for doc in result["source_documents"]:
        print(doc.metadata.get("source", "N/A"))