from langchain_community.vectorstores import FAISS
from langchain_cohere import CohereEmbeddings

def store_in_faiss(chunks, persist_path="../faiss_index"):
    """
    Store document chunks in a FAISS vector store and save to disk.
    I am doing this because FAISS is a fast, local vector database that avoids the rate limits and persistence issues of cloud vector stores.
    """
    embeddings_model = CohereEmbeddings(model="embed-english-v3.0")
    batch_size = 100
    vectorstore = None

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}")
        if vectorstore is None:
            vectorstore = FAISS.from_documents(batch, embeddings_model)
        else:
            vectorstore.add_documents(batch)

    vectorstore.save_local(persist_path)
    print(f"Saved {len(chunks)} chunks to {persist_path}")
    return vectorstore

def load_faiss(persist_path="../faiss_index"):
    """
    Load a FAISS vector store from disk.
    """
    embeddings_model = CohereEmbeddings(model="embed-english-v3.0")
    return FAISS.load_local(persist_path, embeddings_model, allow_dangerous_deserialization=True)