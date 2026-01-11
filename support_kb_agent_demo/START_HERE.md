# START HERE

This is my submission for the Sowiz AI Engineering Internship. A complete, production-ready RAG system that answers questions about documents using AI.

## Quick Start (3 Steps)

### Step 1: Install Dependencies

Navigate to the project directory and install all required packages:

```bash
cd support_kb_agent_demo
pip install -r requirements.txt
```

**What gets installed:**
- `langchain` - LLM framework
- `openai` - OpenAI API integration
- `cohere` - Embeddings API
- `chromadb` - Vector database
- `flask` - Web framework
- `python-dotenv` - Environment variable management
- `pypdf` - PDF processing
- `requests` - HTTP library

### Step 2: Get API Keys

**Cohere API Key (Free):**
1. Go to https://cohere.com/
2. Sign up (free account)
3. Navigate to API keys section
4. Copy your API key
5. Free tier: 1 million API calls/month

**OpenAI API Key (Paid):**
1. Go to https://platform.openai.com/api-keys
2. Sign up or log in
3. Create a new API key
4. Copy the key
5. Pricing: ~$0.002 per query with GPT-3.5-turbo

### Step 3: Set Environment Variables

Create a `.env` file in the `support_kb_agent_demo/` directory:

```bash
# .env file
COHERE_API_KEY=your-cohere-api-key-here
OPENAI_API_KEY=your-openai-api-key-here
```

**Important:** Never commit `.env` to git! It's already in `.gitignore`.

## Run the System

### Option 1: Jupyter Notebook (Recommended for Learning)

The notebook provides step-by-step explanations and is perfect for understanding how the system works:

```bash
jupyter notebook notebooks/support_kb_agent_demo.ipynb
```

**What the notebook does:**
1. Loads your document (PDF, Markdown, or Web URL)
2. Chunks it into manageable pieces (500 tokens each)
3. Creates embeddings using Cohere
4. Stores embeddings in FAISS vector database
5. Expands your query using MCP tools
6. Retrieves relevant chunks
7. Generates answer using OpenAI GPT
8. Displays sources and performance metrics

**Run each cell in order** to see the full pipeline with explanations.

### Option 2: Web Application (Interactive UI)

The web app provides a user-friendly interface for document Q&A:

```bash
python scripts/app.py
```

Then open http://localhost:5000 in your browser.

**Features:**
- Upload documents (PDF, Markdown, or provide URL)
- Ask multiple questions about the same document
- See answers with source citations
- View performance metrics
- Remove document and start fresh

### Option 3: Command Line (For Production)

Run the pipeline directly from the command line:

```bash
python scripts/orchestrate_rag.py \
    --input data/sample.pdf \
    --type pdf \
    --question "What are the key risks mentioned?"
```

**Options:**
- `--input` - Path to document or URL
- `--type` - Document type: `pdf`, `markdown`, or `web`
- `--question` - Your question about the document
- `--no-mcp` - Disable MCP tools (optional, for faster processing)

## How It Works

### The Complete Pipeline

```
1. INGEST
   └─ Load document (PDF, Markdown, or Web)

2. CHUNK
   └─ Split into 500-token chunks with 50-token overlap

3. EMBED & STORE
   └─ Create embeddings with Cohere
   └─ Store in ChromaDB vector database

4. MCP TOOLS (NEW!)
   ├─ Expand query into multiple variants
   ├─ Search with all variants
   └─ Enrich metadata

5. RETRIEVE
   └─ Find top-5 most relevant chunks

6. ANSWER
   └─ Generate answer with OpenAI GPT
   └─ Cite sources

7. OUTPUT
   └─ Return answer + sources + metrics
```

### Why This Approach?

- **Semantic Search:** Finds relevant information even with different wording
- **Source Citation:** Every answer includes where the information came from
- **Performance Tracking:** See exactly how long each step takes
- **MCP Tools:** Automatically enhance retrieval quality
- **Production Ready:** Error handling, logging, and security built-in

## Sample Queries

Try these questions with your documents:

```
"Summarize the main findings"
"What are the key risks mentioned?"
"List all important dates and deadlines"
"What are the recommendations?"
"Who are the stakeholders involved?"
"What is the budget?"
"What are the next steps?"
```

## Expected Output

When you run the system, you'll see:

```
======================================================================
ANSWER:
======================================================================
[Your answer based on the document with sources cited]

======================================================================
SOURCES:
======================================================================
Source 1: sample.pdf (Page 1)
"Relevant excerpt from the document..."

Source 2: sample.pdf (Page 3)
"Another relevant excerpt..."

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

### "API key not found"
- Make sure `.env` file exists in `support_kb_agent_demo/`
- Check that `COHERE_API_KEY` and `GEMINI_API_KEY` are set
- Restart your terminal/notebook after creating `.env`

### "File not found"
- Use absolute paths or relative paths from project root
- Ensure file exists in `support_kb_agent_demo/data/`
- For URLs, make sure they're accessible

### "Connection error"
- Check your internet connection
- Verify API keys are valid
- Check API rate limits (Cohere: 1M/month, Gemini: 60/min)

### "Out of memory"
- Reduce `CHUNK_SIZE` in `.env`
- Process smaller documents
- Clear `chroma_db/` directory to free space

## Next Steps

1. **Run the notebook** to understand the pipeline
2. **Try the web app** for interactive use
3. **Use your own documents** instead of sample.pdf
4. **Customize the questions** for your use case
5. **Explore the code** to understand how it works
6. **Extend with MCP tools** (see MCP_INTEGRATION.md)

## Documentation

- **README.md** - Complete project overview and architecture
- **READY_TO_RUN.md** - Detailed setup and configuration guide
- **MCP_INTEGRATION.md** - MCP tools documentation and examples

---

**You're all set! Set your API keys and run the notebook or web app!**

