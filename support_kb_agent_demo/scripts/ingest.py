import os
from typing import List, Union
try:
	# Try new import paths first (LangChain 0.1+)
	from langchain_community.document_loaders import PyPDFLoader, UnstructuredMarkdownLoader, WebBaseLoader
	from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
except ImportError:
	# Fall back to old import paths (LangChain <0.1)
	from langchain.document_loaders import PyPDFLoader, UnstructuredMarkdownLoader, WebBaseLoader
	from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter

try:
	# Prefer new Document location
	from langchain_core.documents import Document
except ImportError:
	try:
		from langchain.schema import Document  # Older LangChain versions
	except ImportError:
		Document = None

def load_pdf(path: str):
    loader = PyPDFLoader(path)
    return loader.load()

def load_markdown(path: str):
    loader = UnstructuredMarkdownLoader(path)
    return loader.load()

def load_web(path_or_url: str):
	"""
	Load web content from either a URL or a local HTML file.

	- If `path_or_url` starts with http/https, use WebBaseLoader (remote web page).
	- Otherwise, treat it as a local HTML file path and load from disk.
	"""
	# URL case (original behavior)
	if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
		loader = WebBaseLoader(path_or_url)
		return loader.load()

	# Local HTML file case (for uploaded .html/.htm files)
	if Document is None:
		raise ImportError("Document class not available to load local HTML files.")

	if not os.path.exists(path_or_url):
		raise FileNotFoundError(f"File not found: {path_or_url}")

	with open(path_or_url, "r", encoding="utf-8") as f:
		text = f.read()

	return [Document(page_content=text, metadata={"source": path_or_url})]

def chunk_documents(
    docs: List,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    use_markdown_splitter: bool = False
):
    if use_markdown_splitter:
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        return [chunk for doc in docs for chunk in splitter.split_text(doc.page_content)]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", r"(?<=\. )", " ", ""]
    )
    return splitter.split_documents(docs)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Document ingestion and chunking")
    parser.add_argument("--input", type=str, required=True, help="Path to PDF/Markdown file or URL")
    parser.add_argument("--type", type=str, choices=["pdf", "md", "web"], required=True, help="Input type")
    parser.add_argument("--chunk_size", type=int, default=500)
    parser.add_argument("--chunk_overlap", type=int, default=50)
    parser.add_argument("--use_md_splitter", action="store_true")
    args = parser.parse_args()

    if args.type == "pdf":
        docs = load_pdf(args.input)
    elif args.type == "md":
        docs = load_markdown(args.input)
    elif args.type == "web":
        docs = load_web(args.input)
    else:
        raise ValueError("Unsupported type")

    chunks = chunk_documents(
        docs,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        use_markdown_splitter=args.use_md_splitter
    )
    print(f"Loaded {len(docs)} documents, produced {len(chunks)} chunks.")