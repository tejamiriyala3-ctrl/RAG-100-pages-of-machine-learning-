"""Index a PDF or text file into the Chroma "vector db" using Gemini embeddings.

Usage:
    python ingest.py "path/to/file.pdf"

Re-running replaces the existing index, so every stored vector comes from the
same embedding model (must match the one in main.py).
"""
import os
import sys
import time

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

EMBED_MODEL = "gemini-embedding-2"  # keep in sync with main.py
DB_DIR = "vector db"

if len(sys.argv) < 2:
    sys.exit('Usage: python ingest.py "path/to/file.pdf"')
path = sys.argv[1]

if path.lower().endswith(".pdf"):
    from pypdf import PdfReader
    docs = [
        Document(page_content=page.extract_text() or "", metadata={"source": os.path.basename(path), "page": i + 1})
        for i, page in enumerate(PdfReader(path).pages)
    ]
else:
    with open(path, encoding="utf-8") as f:
        docs = [Document(page_content=f.read(), metadata={"source": os.path.basename(path)})]

docs = [d for d in docs if d.page_content.strip()]
chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150).split_documents(docs)
print(f"Loaded {len(docs)} page(s) -> {len(chunks)} chunks")

db = Chroma(
    persist_directory=DB_DIR,
    embedding_function=GoogleGenerativeAIEmbeddings(model=EMBED_MODEL, api_key=os.environ["GOOGLE_API_KEY"]),
)
db.reset_collection()  # drop anything embedded with a different model

# Gemini free tier allows ~100 embedding requests/minute (1 per chunk),
# so send small batches, pause between them, and wait out any 429.
BATCH = 40
PAUSE = 30  # seconds between batches
for i in range(0, len(chunks), BATCH):
    batch = chunks[i:i + BATCH]
    for attempt in range(6):
        try:
            db.add_documents(batch)
            break
        except Exception as e:
            if "429" not in str(e) and "RESOURCE_EXHAUSTED" not in str(e):
                raise
            print(f"  rate limited, waiting 65s (attempt {attempt + 1}/6)...")
            time.sleep(65)
    else:
        sys.exit("Gave up: still rate limited. Try again later.")
    print(f"  indexed {min(i + BATCH, len(chunks))}/{len(chunks)}")
    if i + BATCH < len(chunks):
        time.sleep(PAUSE)
print(f"Done: {len(chunks)} chunks in '{DB_DIR}' with {EMBED_MODEL}")
