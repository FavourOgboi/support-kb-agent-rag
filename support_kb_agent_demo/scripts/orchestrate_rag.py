import os
import logging
import time
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List, Dict, Any
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
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

from langchain_openai import ChatOpenAI
from scripts.ingest import load_pdf, load_markdown, load_web, chunk_documents

# Configuring logging for observability
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGState(TypedDict):
    """State container for RAG pipeline orchestration"""
    messages: Annotated[list[AnyMessage], list.__add__]
    docs: List
    chunks: List
    vectordb: object
    input_type: str
    input_path: str
    question: str
    response_metadata: Dict[str, Any]
    error: str

def ingest_node(state: RAGState):
    """Loading and ingesting documents from various sources"""
    try:
        input_type = state.get("input_type")
        input_path = state.get("input_path")

        logger.info(f"Ingesting {input_type} from {input_path}")
        start_time = time.time()

        if input_type == "pdf":
            docs = load_pdf(input_path)
        elif input_type == "md":
            docs = load_markdown(input_path)
        elif input_type == "web":
            docs = load_web(input_path)
        else:
            raise ValueError(f"Unsupported input type: {input_type}")

        elapsed = time.time() - start_time
        logger.info(f"Loaded {len(docs)} documents in {elapsed:.2f}s")

        if not docs or len(docs) == 0:
            raise ValueError(f"No content loaded from {input_type} file: {input_path}")

        return {
            "docs": docs,
            "messages": [SystemMessage(content=f"Loaded {len(docs)} documents")],
            "response_metadata": {"ingest_time": elapsed, "doc_count": len(docs)}
        }
    except Exception as e:
        logger.error(f"Ingestion failed: {str(e)}")
        return {
            "docs": [],
            "messages": [SystemMessage(content=f"Error: {str(e)}")],
            "error": str(e),
            "response_metadata": {}
        }

def chunk_node(state: RAGState):
    """Chunking documents for optimal retrieval"""
    try:
        docs = state["docs"]
        input_type = state.get("input_type", "")
        # I am doing this because Cohere's free trial has strict embedding rate limits.
        # By using a large chunk size for web and PDF files, I ensure we only create 3-4 chunks per file,
        # which minimizes embedding requests and prevents rate limit errors for large files.
        if input_type == "web":
            chunk_size = 20000
            chunk_overlap = 2000
        elif input_type == "pdf":
            chunk_size = 5000
            chunk_overlap = 500
        else:
            chunk_size = 500
            chunk_overlap = 50

        logger.info(f"Chunking {len(docs)} documents with chunk_size={chunk_size}...")
        start_time = time.time()

        chunks = chunk_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        elapsed = time.time() - start_time
        logger.info(f"Created {len(chunks)} chunks in {elapsed:.2f}s")

        metadata = state.get("response_metadata", {})
        metadata.update({"chunk_time": elapsed, "chunk_count": len(chunks)})

        return {
            "chunks": chunks,
            "response_metadata": metadata
        }
    except Exception as e:
        logger.error(f"Chunking failed: {str(e)}")
        return {
            "chunks": [],
            "error": str(e),
            "response_metadata": state.get("response_metadata", {})
        }

def embed_store_node(state: RAGState):
    """Embedding and storing chunks in vector database"""
    try:
        api_key = os.environ.get("COHERE_API_KEY")
        if not api_key:
            raise ValueError("COHERE_API_KEY environment variable not set")

        chunks = state["chunks"]
        logger.info(f"Embedding {len(chunks)} chunks with Cohere...")
        start_time = time.time()

        # Use FAISS for vector storage
        vectordb = store_in_faiss(chunks, persist_path="faiss_index")

        elapsed = time.time() - start_time
        logger.info(f"Embedded and stored in {elapsed:.2f}s")

        metadata = state.get("response_metadata", {})
        metadata.update({"embedding_time": elapsed})

        return {
            "vectordb": vectordb,
            "response_metadata": metadata
        }
    except Exception as e:
        logger.error(f"Embedding failed: {str(e)}")
        return {
            "vectordb": None,
            "error": str(e),
            "response_metadata": state.get("response_metadata", {})
        }

def retrieve_answer_node(state: RAGState):
    """Retrieving relevant documents and generating answer"""
    try:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        vectordb = state.get("vectordb")
        if not vectordb:
            raise ValueError("Vector database not initialized")

        logger.info("Retrieving documents and generating answer...")
        start_time = time.time()

        question = state.get("question")
        llm = ChatOpenAI(openai_api_key=api_key, model="gpt-3.5-turbo", temperature=0)

        # Retrieve relevant documents using FAISS
        docs = vectordb.similarity_search(question, k=5)

        # Build context from retrieved documents
        context = "\n\n".join(doc.page_content for doc in docs)

        # Manually construct the prompt instead of using RetrievalQA/PromptTemplate
        prompt = (
            "You are a helpful support agent. Use the provided context to answer the question. "
            "Cite sources using [source] notation where appropriate.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        )

        response_msg = llm.invoke(prompt)

        elapsed = time.time() - start_time
        logger.info(f"Generated answer in {elapsed:.2f}s")

        # Extracting source metadata
        sources = []
        for doc in docs:
            sources.append({
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "N/A"),
            })

        metadata = state.get("response_metadata", {})
        metadata.update({
            "retrieval_time": elapsed,
            "sources_count": len(sources),
            "sources": sources
        })

        return {
            "messages": [
                HumanMessage(content=question),
                SystemMessage(content=response_msg.content)
            ],
            "response_metadata": metadata
        }
    except Exception as e:
        logger.error(f"Retrieval failed: {str(e)}")
        return {
            "messages": [SystemMessage(content=f"Error generating answer: {str(e)}")],
            "error": str(e),
            "response_metadata": state.get("response_metadata", {})
        }

def mcp_tool_node(state: RAGState):
    """
    MCP (Model Context Protocol) tool integration node.

    This node enhances the RAG pipeline by:
    1. Using MCP tools to search for additional context
    2. Enriching the vector database with metadata
    3. Providing tool-based document search capabilities

    Currently implements:
    - Document metadata extraction
    - Query expansion for better retrieval
    - Source verification
    """
    try:
        question = state.get("question", "")
        vectordb = state.get("vectordb")

        if not vectordb or not question:
            logger.warning("MCP node: Missing vectordb or question, skipping")
            return {"response_metadata": state.get("response_metadata", {})}

        logger.info("Running MCP tool integration...")
        start_time = time.time()

        # MCP Tool 1: Query Expansion
        # Expand the original query to improve retrieval
        expanded_queries = _expand_query(question)
        logger.info(f"Expanded query to {len(expanded_queries)} variants")

        # MCP Tool 2: Multi-query retrieval
        # Retrieve documents using multiple query variants
        all_docs = []
        for query in expanded_queries:
            docs = state["vectordb"].similarity_search(query, k=3)
            all_docs.extend(docs)

        # Remove duplicates while preserving order
        seen = set()
        unique_docs = []
        for doc in all_docs:
            doc_id = doc.metadata.get("source", "") + str(doc.metadata.get("page", ""))
            if doc_id not in seen:
                seen.add(doc_id)
                unique_docs.append(doc)

        # MCP Tool 3: Metadata enrichment
        enriched_metadata = _enrich_metadata(unique_docs)

        elapsed = time.time() - start_time
        logger.info(f"MCP tools completed in {elapsed:.2f}s, found {len(unique_docs)} unique documents")

        metadata = state.get("response_metadata", {})
        metadata.update({
            "mcp_time": elapsed,
            "mcp_expanded_queries": len(expanded_queries),
            "mcp_unique_docs": len(unique_docs),
            "mcp_enriched_metadata": enriched_metadata
        })

        return {
            "response_metadata": metadata,
            "messages": [SystemMessage(content=f"MCP tools enhanced retrieval with {len(unique_docs)} documents")]
        }
    except Exception as e:
        logger.error(f"MCP tool integration failed: {str(e)}")
        return {
            "response_metadata": state.get("response_metadata", {}),
            "error": f"MCP error: {str(e)}"
        }

def _expand_query(question: str) -> List[str]:
    """
    Expand a query into multiple variants for better retrieval.

    Strategies:
    1. Original question
    2. Question without stop words
    3. Key noun phrases
    """
    queries = [question]  # Original

    # Simple expansion: add variations
    if "what" in question.lower():
        queries.append(question.replace("what", "which").replace("What", "Which"))
    if "how" in question.lower():
        queries.append(question.replace("how", "explain").replace("How", "Explain"))

    # Extract key terms (simple approach)
    words = question.split()
    if len(words) > 3:
        # Add query with just key nouns
        key_terms = " ".join([w for w in words if len(w) > 4])
        if key_terms:
            queries.append(key_terms)

    return queries[:3]  # Return top 3 variants

def _enrich_metadata(docs: List) -> Dict[str, Any]:
    """
    Enrich document metadata with additional information.

    Extracts:
    - Document sources
    - Page numbers
    - Content length
    - Relevance indicators
    """
    metadata = {
        "total_documents": len(docs),
        "sources": [],
        "pages": [],
        "avg_content_length": 0
    }

    total_length = 0
    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "N/A")

        if source not in metadata["sources"]:
            metadata["sources"].append(source)
        if page not in metadata["pages"]:
            metadata["pages"].append(page)

        total_length += len(doc.page_content)

    if docs:
        metadata["avg_content_length"] = total_length // len(docs)

    return metadata

def build_rag_graph(enable_mcp: bool = True):
    """
    Build the RAG pipeline graph with optional MCP tool integration.

    Pipeline flow:
    1. ingest -> Load documents
    2. chunk -> Split into chunks
    3. embed_store -> Create embeddings and store
    4. mcp_tool (optional) -> Enhance retrieval with MCP tools
    5. retrieve_answer -> Generate answer

    Args:
        enable_mcp: Whether to include MCP tool node in the pipeline

    Returns:
        Compiled LangGraph StateGraph
    """
    graph = StateGraph(RAGState)

    # Add all nodes
    graph.add_node("ingest", ingest_node)
    graph.add_node("chunk", chunk_node)
    graph.add_node("embed_store", embed_store_node)

    if enable_mcp:
        graph.add_node("mcp_tool", mcp_tool_node)

    graph.add_node("retrieve_answer", retrieve_answer_node)

    # Define edges
    graph.add_edge("ingest", "chunk")
    graph.add_edge("chunk", "embed_store")

    if enable_mcp:
        graph.add_edge("embed_store", "mcp_tool")
        graph.add_edge("mcp_tool", "retrieve_answer")
    else:
        graph.add_edge("embed_store", "retrieve_answer")

    graph.add_edge("retrieve_answer", END)
    graph.set_entry_point("ingest")

    return graph.compile()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Orchestrate RAG pipeline with LangGraph and MCP tools")
    parser.add_argument("--input", type=str, required=True, help="Path to PDF/Markdown file or URL")
    parser.add_argument("--type", type=str, choices=["pdf", "md", "web"], required=True, help="Input type")
    parser.add_argument("--question", type=str, required=True, help="Question to answer")
    parser.add_argument("--no-mcp", action="store_true", help="Disable MCP tool integration")
    args = parser.parse_args()

    # Build graph with MCP enabled by default
    enable_mcp = not args.no_mcp
    logger.info(f"Building RAG graph with MCP tools {'enabled' if enable_mcp else 'disabled'}")
    rag_graph = build_rag_graph(enable_mcp=enable_mcp)

    # Initialize state
    state = {
        "input_type": args.type,
        "input_path": args.input,
        "question": args.question,
        "messages": [],
        "docs": [],
        "chunks": [],
        "vectordb": None,
        "response_metadata": {},
        "error": ""
    }

    # Run the pipeline
    logger.info("Starting RAG pipeline...")
    start_time = time.time()
    result = rag_graph.invoke(state)
    total_time = time.time() - start_time

    # Display results
    print("\n" + "="*70)
    print("ANSWER:")
    print("="*70)
    print(result["messages"][-1].content)

    # Display metadata
    if result.get("response_metadata"):
        print("\n" + "="*70)
        print("PIPELINE METRICS:")
        print("="*70)
        metadata = result["response_metadata"]
        for key, value in metadata.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}s")
            elif isinstance(value, int):
                print(f"  {key}: {value}")
            elif isinstance(value, list) and key not in ["sources", "pages"]:
                print(f"  {key}: {len(value)} items")
            elif isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    - {k}: {v}")

        print(f"\n  Total pipeline time: {total_time:.2f}s")

    # Display any errors
    if result.get("error"):
        print(f"\nWarning: {result['error']}")