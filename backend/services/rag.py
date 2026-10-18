
import os
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from google import genai
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

load_dotenv()

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.6-flash"
)

# Sentence Transformers creates local text embeddings.
embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# Gemini is used only for answer generation.
api_key = os.getenv("GEMINI_API_KEY")
gemini_client = genai.Client(api_key=api_key) if api_key else None

# In-memory document index for the prototype.
chunks = []
metadata = []
vector_index = None


def split_text(text: str, chunk_size=900, overlap=150):
    """Split text into overlapping character-based chunks."""

    text = " ".join(text.split())
    result = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        part = text[start:end].strip()

        if part:
            result.append(part)

        if end == len(text):
            break

        start = end - overlap

    return result


def index_pdf(pdf_path: str):
    """Extract PDF text and add its chunks to FAISS."""

    global vector_index

    reader = PdfReader(pdf_path)
    added_chunks = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""

        for part in split_text(page_text):
            added_chunks.append({
                "text": part,
                "source": Path(pdf_path).name,
                "page": page_number,
            })

    if not added_chunks:
        raise ValueError(
            "No extractable text was found. "
            "Scanned PDFs may require OCR."
        )

    texts = [item["text"] for item in added_chunks]

    vectors = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    if vector_index is None:
        vector_index = faiss.IndexFlatIP(vectors.shape[1])

    vector_index.add(vectors)

    chunks.extend(texts)
    metadata.extend([
        {
            "source": item["source"],
            "page": item["page"],
        }
        for item in added_chunks
    ])

    return len(added_chunks)


def search_documents(question: str, top_k=4):
    """Retrieve the most relevant indexed passages."""

    if vector_index is None or vector_index.ntotal == 0:
        return []

    query_vector = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    count = min(top_k, vector_index.ntotal)
    scores, indices = vector_index.search(query_vector, count)

    results = []

    for score, index in zip(scores[0], indices[0]):
        if index < 0:
            continue

        results.append({
            "text": chunks[index],
            "source": metadata[index]["source"],
            "page": metadata[index]["page"],
            "similarity": round(float(score), 4),
        })

    return results


def ask_maritime_question(question: str):
    """Answer using retrieved context and cite the sources."""

    if not question.strip():
        raise ValueError("Please enter a question.")

    if gemini_client is None:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured in the .env file."
        )

    results = search_documents(question)

    if not results:
        return {
            "answer": (
                "No indexed document passages are available. "
                "Upload a text-based maritime safety PDF first."
            ),
            "sources": [],
        }

    context = "\n\n".join(
        (
            f"[Source: {item['source']}, "
            f"page {item['page']}]\n{item['text']}"
        )
        for item in results
    )

    prompt = f"""
You are MarineGuard AI, a maritime safety information assistant.

Answer the user's question using only the supplied document context.
Do not invent regulations, facts, or source references.
If the context does not contain the answer, say so clearly.
Distinguish document-based information from any uncertainty.
Do not present your answer as an official operational instruction.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}
"""

    response = gemini_client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )

    return {
        "answer": response.text or "No answer was generated.",
        "sources": [
            {
                "source": item["source"],
                "page": item["page"],
                "similarity": item["similarity"],
            }
            for item in results
        ],
    }


def load_existing_pdfs():
    """Rebuild the in-memory index from saved PDFs."""

    for pdf_path in sorted(UPLOAD_DIR.glob("*.pdf")):
        try:
            index_pdf(str(pdf_path))
            print(f"Indexed existing PDF: {pdf_path.name}")
        except Exception as exc:
            print(f"Could not index {pdf_path.name}: {exc}")
