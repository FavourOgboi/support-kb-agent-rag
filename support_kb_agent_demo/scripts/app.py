from flask import Flask, render_template, request, jsonify
import os
import sys
import logging
from dotenv import load_dotenv

# Loading environment variables from .env file
env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
if os.path.exists(env_path):
    load_dotenv(env_path)
else:
    print(f"Warning: .env file not found at {env_path}")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Configuring logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up Flask with correct template and static folders
template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "templates"))
static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "static"))
app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
app.config['JSON_SORT_KEYS'] = False

# Global cache for RAG graph
_rag_graph_cache = None

def get_rag_graph():
    """Lazy load the RAG graph to avoid startup errors"""
    global _rag_graph_cache
    if _rag_graph_cache is None:
        logger.info("Loading RAG graph...")
        try:
            from scripts.orchestrate_rag import build_rag_graph
            _rag_graph_cache = build_rag_graph()
            logger.info("RAG graph loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load RAG graph: {e}", exc_info=True)
            raise
    return _rag_graph_cache

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({"status": "ok", "message": "Support KB Agent is running"}), 200

@app.route("/", methods=["GET", "POST"])
def index():
    """Main page handling document queries"""
    answer = None
    sources = []
    metadata = {}
    error = None

    if request.method == "POST":
        try:
            input_type = request.form.get("input_type", "pdf")
            input_path = request.form.get("input_path", "")
            question = request.form.get("question", "")

            # Handle file upload
            uploaded_file = request.files.get("input_file")
            if uploaded_file and uploaded_file.filename:
                # Validate file type
                allowed_types = {
                    "pdf": [".pdf"],
                    "md": [".md"],
                    "web": [".html", ".htm", ".txt"]
                }
                ext = os.path.splitext(uploaded_file.filename)[1].lower()
                if ext not in allowed_types.get(input_type, []):
                    error = f"Invalid file type for {input_type.upper()}. Please upload a {', '.join(allowed_types[input_type])} file."
                    return render_template("index.html", answer=answer, sources=sources, metadata=metadata, error=error)
                # Save file to data directory
                data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
                os.makedirs(data_dir, exist_ok=True)
                save_path = os.path.join(data_dir, uploaded_file.filename)
                uploaded_file.save(save_path)
                input_path = save_path

            if (not input_path or input_path.strip() == "") or not question:
                error = "Please provide a file (or path/URL) and a question"
                return render_template("index.html", answer=answer, sources=sources, metadata=metadata, error=error)

            logger.info(f"Processing query: {question[:50]}...")

            rag_graph = get_rag_graph()
            state = {
                "input_type": input_type,
                "input_path": input_path,
                "question": question,
                "messages": [],
                "docs": [],
                "chunks": [],
                "vectordb": None,
                "response_metadata": {},
                "error": ""
            }

            result = rag_graph.invoke(state)

            if result.get("error"):
                error = f"{result['error']}"
                logger.error(error)
            else:
                answer = result["messages"][-1].content
                metadata = result.get("response_metadata", {})
                sources = metadata.get("sources", [])
                logger.info("Query processed successfully")

        except Exception as e:
            error = f"Error processing query: {str(e)}"
            logger.error(error)

    return render_template("index.html", answer=answer, sources=sources, metadata=metadata, error=error)

@app.route("/api/query", methods=["POST"])
def api_query():
    """API endpoint for programmatic access"""
    try:
        data = request.get_json()
        input_type = data.get("input_type", "pdf")
        input_path = data.get("input_path", "")
        question = data.get("question", "")

        if not input_path or not question:
            return jsonify({"error": "Missing input_path or question"}), 400

        rag_graph = get_rag_graph()
        state = {
            "input_type": input_type,
            "input_path": input_path,
            "question": question,
            "messages": [],
            "docs": [],
            "chunks": [],
            "vectordb": None,
            "response_metadata": {},
            "error": ""
        }

        result = rag_graph.invoke(state)

        return jsonify({
            "answer": result["messages"][-1].content if result["messages"] else "",
            "metadata": result.get("response_metadata", {}),
            "error": result.get("error", "")
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("=" * 60)
    print("Starting Support KB Agent Demo")
    print("=" * 60)
    print("Flask app running on http://localhost:5000")
    print("Health check: http://localhost:5000/health")
    print("=" * 60)
    app.run(debug=False, host="0.0.0.0", port=5000, use_reloader=False)