# MCP Tool Integration Guide

## Overview

The Support KB Agent now includes **fully integrated MCP (Model Context Protocol) tools** that automatically enhance the RAG pipeline without requiring any manual setup.

## What is MCP?

MCP is a protocol for integrating external tools and services into AI applications. In this project, MCP tools enhance document retrieval by:

1. **Query Expansion** - Converts a single query into multiple variants
2. **Multi-Query Retrieval** - Searches using all query variants
3. **Metadata Enrichment** - Extracts and enriches document metadata

## How It Works

### Pipeline Flow (with MCP enabled)

```
Document Input
    ↓
[ingest_node] → Load documents
    ↓
[chunk_node] → Split into chunks
    ↓
[embed_store_node] → Create embeddings & store
    ↓
[mcp_tool_node] ← NEW! Enhance retrieval
    ├─ Expand query variants
    ├─ Multi-query retrieval
    └─ Enrich metadata
    ↓
[retrieve_answer_node] → Generate answer
    ↓
Answer with Sources
```

## MCP Tools Implemented

### 1. Query Expansion (`_expand_query`)

Converts a single question into multiple variants:

```python
Original: "What are the key risks?"
Variants:
  - "What are the key risks?"
  - "Which are the key risks?"
  - "key risks"
```

**Benefits:**
- Captures different phrasings
- Improves recall
- Finds documents using alternative terminology

### 2. Multi-Query Retrieval

Searches the vector database using all expanded queries:

```python
for query in expanded_queries:
    docs = vectordb.similarity_search(query, k=3)
```

**Benefits:**
- Retrieves more relevant documents
- Reduces false negatives
- Combines results from multiple search angles

### 3. Metadata Enrichment (`_enrich_metadata`)

Extracts and organizes document metadata:

```python
{
    "total_documents": 5,
    "sources": ["sample.pdf"],
    "pages": [1, 2, 3],
    "avg_content_length": 450
}
```

**Benefits:**
- Provides document provenance
- Enables source tracking
- Supports document filtering

## Usage

### Enable MCP (Default)

```python
from scripts.orchestrate_rag import build_rag_graph

# MCP is enabled by default
rag_graph = build_rag_graph(enable_mcp=True)
```

### Disable MCP (if needed)

```python
# Skip MCP tools for faster processing
rag_graph = build_rag_graph(enable_mcp=False)
```

### Command Line

```bash
# With MCP (default)
python scripts/orchestrate_rag.py \
    --input data/sample.pdf \
    --type pdf \
    --question "What are the key risks?"

# Without MCP
python scripts/orchestrate_rag.py \
    --input data/sample.pdf \
    --type pdf \
    --question "What are the key risks?" \
    --no-mcp
```

## Performance Metrics

The pipeline tracks MCP performance:

```
PIPELINE METRICS:
  ingest_time: 0.45s
  chunk_time: 0.12s
  embedding_time: 2.34s
  mcp_time: 0.28s          ← MCP processing time
  mcp_expanded_queries: 3  ← Number of query variants
  mcp_unique_docs: 8       ← Documents found
  retrieval_time: 1.56s
  Total pipeline time: 4.75s
```

## Extending MCP Tools

To add new MCP tools, extend the `mcp_tool_node` function:

```python
def mcp_tool_node(state: RAGState):
    # ... existing code ...
    
    # Add your custom tool
    custom_results = my_custom_tool(question, vectordb)
    
    # Update metadata
    metadata.update({"custom_tool_results": custom_results})
    
    return {"response_metadata": metadata}
```

## Key Features

✓ **Automatic** - No configuration needed
✓ **Transparent** - Works seamlessly in the pipeline
✓ **Observable** - Metrics tracked and reported
✓ **Extensible** - Easy to add new tools
✓ **Efficient** - Minimal performance overhead

## Summary

MCP tools are now fully integrated and enabled by default. They automatically enhance retrieval quality without requiring any manual setup. Just provide your API keys and run!

