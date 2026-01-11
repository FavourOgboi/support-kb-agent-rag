# Ready to Run! 🚀

## What You Need to Do

**That's it! Just 2 steps:**

### Step 1: Get API Keys

1. **Cohere API Key** (Free tier available)
   - Go to https://cohere.com/
   - Sign up and create an API key
   - Copy the key

2. **OpenAI API Key** (Paid)
   - Go to https://platform.openai.com/api-keys
   - Sign up and create an API key
   - Copy the key

### Step 2: Set Environment Variables

**Option A: Using .env file (Recommended)**

```bash
# Create .env file in the project root
COHERE_API_KEY=your-cohere-key-here
OPENAI_API_KEY=your-openai-key-here
```

**Option B: Using system environment variables**

Windows (PowerShell):
```powershell
$env:COHERE_API_KEY = "your-cohere-key"
$env:OPENAI_API_KEY = "your-openai-key"
```

Linux/Mac:
```bash
export COHERE_API_KEY="your-cohere-key"
export OPENAI_API_KEY="your-openai-key"
```

## Run the Pipeline

### Option 1: Jupyter Notebook (Recommended for Learning)

```bash
cd support_kb_agent_demo
jupyter notebook notebooks/support_kb_agent_demo.ipynb
```

Then run each cell in order. The notebook includes:
- Step-by-step explanations
- Working code examples
- Performance metrics
- MCP tool integration showcase

### Option 2: Command Line (For Production)

```bash
python scripts/orchestrate_rag.py \
    --input data/sample.pdf \
    --type pdf \
    --question "What are the key risks mentioned?"
```

### Option 3: Python Script

```python
from scripts.orchestrate_rag import build_rag_graph

# Build pipeline with MCP tools enabled (default)
rag_graph = build_rag_graph(enable_mcp=True)

# Initialize state
state = {
    "input_type": "pdf",
    "input_path": "data/sample.pdf",
    "question": "What are the key risks?",
    "messages": [],
    "docs": [],
    "chunks": [],
    "vectordb": None,
    "response_metadata": {},
    "error": ""
}

# Run the pipeline
result = rag_graph.invoke(state)

# Get the answer
print(result["messages"][-1].content)
```

## What Happens Automatically

Once you run it, the system automatically:

1. ✅ Loads your document (PDF, Markdown, or Web)
2. ✅ Chunks it into manageable pieces
3. ✅ Creates embeddings using Cohere
4. ✅ Stores in ChromaDB vector database
5. ✅ **Runs MCP tools** (query expansion, multi-query retrieval)
6. ✅ Retrieves relevant chunks
7. ✅ Generates answer using OpenAI
8. ✅ Returns answer with sources and metrics

**No manual setup needed!** Everything is integrated and automated.

## Expected Output

```
======================================================================
ANSWER:
======================================================================
[Your answer based on the document]

======================================================================
PIPELINE METRICS:
======================================================================
  ingest_time: 0.45s
  chunk_time: 0.12s
  embedding_time: 2.34s
  mcp_time: 0.28s
  mcp_expanded_queries: 3
  mcp_unique_docs: 8
  retrieval_time: 1.56s
  
  Total pipeline time: 4.75s
```

## Troubleshooting

**"API key not found"**
- Make sure you set COHERE_API_KEY and OPENAI_API_KEY
- Check that .env file is in the project root
- Restart your terminal/notebook

**"File not found"**
- Make sure the document path is correct
- Use absolute paths if relative paths don't work

**"Connection error"**
- Check your internet connection
- Verify API keys are valid
- Check API rate limits

## Next Steps

1. Run the notebook to understand the pipeline
2. Try with your own documents
3. Customize the question
4. Explore the code to understand how it works
5. Extend with your own MCP tools (see MCP_INTEGRATION.md)

## Documentation

- **Quick Start**: QUICK_START.md
- **MCP Tools**: MCP_INTEGRATION.md
- **Full Implementation**: COMPLETE_IMPLEMENTATION.md
- **Architecture**: README.md

---

**Everything is ready! Just set your API keys and run!** 🎉

