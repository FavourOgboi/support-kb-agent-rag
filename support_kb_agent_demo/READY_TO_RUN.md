# Ready to Run!

This is my submission for the Sowiz AI Engineering Internship. Everything you need to get the Support KB Agent running in minutes.

## What You Need to Do

**Just 2 steps to get started:**

### Step 1: Get API Keys

#### Cohere API Key (Free Tier Available)

1. Go to https://cohere.com/
2. Click "Sign Up" (free account)
3. Verify your email
4. Go to Dashboard → API Keys
5. Click "Create API Key"
6. Copy the key and save it

**Free Tier Benefits:**
- 1 million API calls per month
- Perfect for development and testing
- No credit card required

#### OpenAI API Key (Paid)

1. Go to https://platform.openai.com/api-keys
2. Sign up or log in with your account
3. Click "Create new secret key"
4. Copy the key and save it securely
5. Add billing information to your account

**Pricing:**
- GPT-3.5-turbo: ~$0.002 per query
- Very affordable for production use
- Pay only for what you use

### Step 2: Set Environment Variables

#### Option A: Using .env file (Recommended)

Create a `.env` file in the `support_kb_agent_demo/` directory:

```bash
# support_kb_agent_demo/.env
COHERE_API_KEY=your-cohere-api-key-here
OPENAI_API_KEY=your-openai-api-key-here
```

**Why this method?**
- Secure (never committed to git)
- Easy to manage
- Works across all platforms
- Automatically loaded by python-dotenv

#### Option B: Using System Environment Variables

**Windows (PowerShell):**
```powershell
$env:COHERE_API_KEY = "your-cohere-key"
$env:OPENAI_API_KEY = "your-openai-key"
```

**Windows (Command Prompt):**
```cmd
set COHERE_API_KEY=your-cohere-key
set OPENAI_API_KEY=your-openai-key
```

**Linux/Mac (Bash):**
```bash
export COHERE_API_KEY="your-cohere-key"
export OPENAI_API_KEY="your-openai-key"
```

**Linux/Mac (Permanent - add to ~/.bashrc or ~/.zshrc):**
```bash
echo 'export COHERE_API_KEY="your-cohere-key"' >> ~/.bashrc
echo 'export OPENAI_API_KEY="your-openai-key"' >> ~/.bashrc
source ~/.bashrc
```

## Run the Pipeline

### Option 1: Jupyter Notebook (Recommended for Learning)

The notebook is perfect for understanding how the system works step-by-step:

```bash
cd support_kb_agent_demo
jupyter notebook notebooks/support_kb_agent_demo.ipynb
```

**What you'll learn:**
- How documents are loaded and processed
- How chunking works and why it matters
- How embeddings capture semantic meaning
- How the vector database stores and retrieves information
- How MCP tools enhance retrieval
- How the LLM generates answers with sources
- Real-time performance metrics

**Run each cell in order** to see the complete pipeline with explanations.

### Option 2: Web Application (Interactive UI)

The web app provides a user-friendly interface for document Q&A:

```bash
cd support_kb_agent_demo
python scripts/app.py
```

Then open http://localhost:5000 in your browser.

**Features:**
- Upload documents (PDF, Markdown, or provide URL)
- Ask multiple questions about the same document
- Session-based persistence (document stays loaded)
- See answers with source citations
- View performance metrics for each query
- Remove document and start fresh
- Real-time loading feedback

**How to use:**
1. Upload a document (PDF, Markdown, or URL)
2. Type your question
3. Click "Ask Agent"
4. View the answer with sources and metrics
5. Ask more questions about the same document
6. Click "Remove Document" to start over

### Option 3: Command Line (For Production)

Run the pipeline directly from the command line:

```bash
cd support_kb_agent_demo
python scripts/orchestrate_rag.py \
    --input data/sample.pdf \
    --type pdf \
    --question "What are the key risks mentioned?"
```

**Command-line options:**
- `--input` - Path to document or URL (required)
- `--type` - Document type: `pdf`, `markdown`, or `web` (required)
- `--question` - Your question about the document (required)
- `--no-mcp` - Disable MCP tools for faster processing (optional)

**Examples:**

```bash
# PDF file
python scripts/orchestrate_rag.py \
    --input data/sample.pdf \
    --type pdf \
    --question "What are the main findings?"

# Markdown file
python scripts/orchestrate_rag.py \
    --input data/document.md \
    --type markdown \
    --question "List all recommendations"

# Web URL
python scripts/orchestrate_rag.py \
    --input "https://example.com/article" \
    --type web \
    --question "What is the main topic?"

# Without MCP tools (faster)
python scripts/orchestrate_rag.py \
    --input data/sample.pdf \
    --type pdf \
    --question "What are the risks?" \
    --no-mcp
```

### Option 4: Python Script (For Integration)

Use the RAG pipeline in your own Python code:

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
answer = result["messages"][-1].content
print(f"Answer: {answer}")

# Get performance metrics
metrics = result["response_metadata"]
print(f"Pipeline time: {metrics.get('total_time', 'N/A')}s")
```

**Advanced usage:**

```python
# Disable MCP tools for faster processing
rag_graph = build_rag_graph(enable_mcp=False)

# Use custom chunk size
import os
os.environ["CHUNK_SIZE"] = "1000"

# Use custom retrieval k
os.environ["TOP_K_RETRIEVAL"] = "10"
```

## What Happens Automatically

Once you run the system, it automatically:

1. **Loads your document** (PDF, Markdown, or Web)
   - Extracts text content
   - Validates file format
   - Handles errors gracefully

2. **Chunks it into pieces** (500 tokens each)
   - Preserves semantic boundaries
   - Maintains 50-token overlap
   - Keeps related content together

3. **Creates embeddings** using Cohere
   - Converts text to 1024-dimensional vectors
   - Captures semantic meaning
   - Enables similarity search

4. **Stores in FAISS** vector database
   - Persistent storage to disk
   - Fast similarity search
   - Batch processing for efficiency

5. **Runs MCP tools** (automatic enhancement)
   - Expands your query into variants
   - Searches with all variants
   - Combines results for better recall

6. **Retrieves relevant chunks** (top-5)
   - Finds most similar documents
   - Ranks by relevance
   - Preserves source information

7. **Generates answer** using OpenAI GPT
   - Reads retrieved context
   - Generates coherent response
   - Cites sources

8. **Returns complete output**
   - Answer with source citations
   - Performance metrics
   - Metadata and statistics

**No manual setup needed!** Everything is integrated and automated.

## Expected Output

### Console Output

```
======================================================================
ANSWER:
======================================================================
Based on the document, the key risks mentioned are:

1. Market volatility - The document highlights potential market
   fluctuations that could impact revenue.
2. Regulatory changes - New regulations could affect operations.
3. Resource constraints - Limited budget for expansion.

[Source: sample.pdf, Page 2]

======================================================================
SOURCES:
======================================================================
Source 1: sample.pdf (Page 2)
"Market volatility is a significant risk that could impact our
quarterly revenue by up to 15%..."

Source 2: sample.pdf (Page 3)
"Regulatory changes in the industry are expected within the next
fiscal year..."

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

### Web App Output

The web app displays:
- Answer with formatted text
- Source snippets with document references
- Performance metrics in a metrics panel
- Loading states and error messages
- Session information (document loaded, etc.)

## Troubleshooting

### "API key not found"
**Problem:** System can't find your API keys

**Solutions:**
- Make sure `.env` file exists in `support_kb_agent_demo/`
- Check that `COHERE_API_KEY` and `OPENAI_API_KEY` are set
- Verify there are no typos in the variable names
- Restart your terminal/notebook after creating `.env`
- Try using system environment variables instead

**Verify your keys:**
```bash
# Linux/Mac
echo $COHERE_API_KEY
echo $OPENAI_API_KEY

# Windows PowerShell
$env:COHERE_API_KEY
$env:OPENAI_API_KEY
```

### "File not found"
**Problem:** System can't find your document

**Solutions:**
- Use absolute paths: `/full/path/to/document.pdf`
- Or relative paths from project root: `data/sample.pdf`
- Ensure file exists: `ls data/sample.pdf` (Linux/Mac) or `dir data\sample.pdf` (Windows)
- Check file permissions (should be readable)

### "Connection error"
**Problem:** Can't connect to API services

**Solutions:**
- Check your internet connection
- Verify API keys are valid (test on API provider's website)
- Check API rate limits:
  - Cohere: 1 million calls/month (free tier)
  - OpenAI: Check your account usage at platform.openai.com
- Ensure you have billing set up on your OpenAI account
- Try again after a few seconds (temporary network issue)

### "Out of memory"
**Problem:** System runs out of memory

**Solutions:**
- Reduce `CHUNK_SIZE` in `.env`: `CHUNK_SIZE=250`
- Process smaller documents
- Clear `chroma_db/` directory: `rm -rf chroma_db/`
- Close other applications to free memory

### "Slow performance"
**Problem:** Pipeline takes too long

**Solutions:**
- Disable MCP tools: `--no-mcp` flag
- Reduce `TOP_K_RETRIEVAL` in `.env`: `TOP_K_RETRIEVAL=3`
- Use smaller documents
- Check internet connection (API calls may be slow)

## Configuration Options

You can customize the system by setting environment variables in `.env`:

```bash
# Chunking
CHUNK_SIZE=500              # Tokens per chunk (default: 500)
CHUNK_OVERLAP=50            # Token overlap (default: 50)

# Retrieval
TOP_K_RETRIEVAL=5           # Number of chunks to retrieve (default: 5)

# LLM
TEMPERATURE=0               # Response randomness (default: 0)
MAX_TOKENS=1000             # Max response length (default: 1000)

# MCP Tools
ENABLE_MCP=true             # Enable MCP tools (default: true)
```

## Next Steps

1. **Run the notebook** to understand the pipeline
2. **Try the web app** for interactive use
3. **Use your own documents** instead of sample.pdf
4. **Customize the questions** for your use case
5. **Explore the code** to understand how it works
6. **Extend with MCP tools** (see MCP_INTEGRATION.md)
7. **Deploy to production** (see README.md for deployment options)

## Documentation

- **START_HERE.md** - Quick start guide
- **README.md** - Complete project overview and architecture
- **MCP_INTEGRATION.md** - MCP tools documentation and examples

---

**Everything is ready! Just set your API keys and run!**

