# START HERE 🚀

## Quick Start (3 Steps)

### Step 1: Install Dependencies
```bash
cd support_kb_agent_demo
pip install -r requirements.txt
```

### Step 2: Get API Keys

**Cohere**: https://cohere.com/ (Free tier available)
**Google Gemini**: https://ai.google.dev (Free tier available)

### Step 3: Set Environment Variables

Create `.env` file:
```
COHERE_API_KEY=your-key
GEMINI_API_KEY=your-key
```

## Run the Notebook

```bash
jupyter notebook notebooks/support_kb_agent_demo.ipynb
```

Run each cell in order. The notebook will:
1. Load your document
2. Chunk it
3. Create embeddings
4. Store in vector database
5. Answer your questions

## Pipeline

1. Load document (PDF, Markdown, or Web)
2. Chunk into pieces
3. Create embeddings with Cohere
4. Store in ChromaDB
5. Retrieve relevant chunks
6. Generate answer with Google Gemini
7. Return answer with sources

## Documentation

- **README.md** - Project overview
- **READY_TO_RUN.md** - Setup guide
- **MCP_INTEGRATION.md** - MCP tool details

---

**Set API keys and run the notebook!**

