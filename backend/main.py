
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from backend.database import (
    add_incident,
    initialize_database,
    list_incidents,
)
from backend.services.classifier import classify_incident
from backend.services.rag import (
    UPLOAD_DIR,
    ask_maritime_question,
    index_pdf,
    load_existing_pdfs,
)

app = FastAPI(
    title="MarineGuard AI API",
    description="Maritime safety information and incident analysis",
    version="1.0.0",
)


@app.on_event("startup")
def startup():
    initialize_database()
    load_existing_pdfs()


class IncidentRequest(BaseModel):
    description: str = Field(min_length=10, max_length=5000)
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)
    location: str | None = Field(default=None, max_length=200)


class QuestionRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)


@app.get("/health")
def health():
    return {"status": "ok", "application": "MarineGuard AI"}


@app.get("/incidents")
def get_incidents():
    return {"incidents": list_incidents()}


@app.post("/incidents")
def create_incident(request: IncidentRequest):
    try:
        prediction = classify_incident(request.description)

        incident_id = add_incident(
            description=request.description,
            category=prediction["category"],
            confidence=prediction["confidence"],
            latitude=request.latitude,
            longitude=request.longitude,
            location=request.location,
        )

        return {
            "id": incident_id,
            **prediction,
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc


@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )

    # Limit the initial prototype to 10 MB per upload.
    content = await file.read(10 * 1024 * 1024 + 1)

    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="The PDF exceeds the 10 MB limit.",
        )

    filename = f"{uuid4().hex}.pdf"
    file_path = UPLOAD_DIR / filename

    try:
        file_path.write_bytes(content)
        number_of_chunks = index_pdf(str(file_path))

        return {
            "message": "PDF indexed successfully.",
            "document_id": filename,
            "chunks_indexed": number_of_chunks,
        }

    except Exception as exc:
        file_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail=f"Could not process PDF: {exc}",
        ) from exc


@app.post("/ask")
def ask_question(request: QuestionRequest):
    try:
        return ask_maritime_question(request.question)

    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="The assistant could not process the question.",
        ) from exc
