"""FastAPI application entrypoint for the RAG research-paper assistant."""

import io
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from .config import settings
from .llm import GroqLLM
from .pdf_parser import PDFParseError, parse_pdf
from .retriever import ColPaliRetriever
from .schemas import AskRequest, HealthResponse, UploadResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global singletons (initialized lazily to keep startup fast)
# ---------------------------------------------------------------------------
retriever: ColPaliRetriever | None = None
llm: GroqLLM | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: initialize shared services."""
    global retriever, llm
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    retriever = ColPaliRetriever()
    try:
        llm = GroqLLM()
    except RuntimeError as exc:
        logger.warning("LLM not initialized: %s", exc)
        llm = None
    yield


app = FastAPI(
    title="Research Paper RAG Assistant",
    description="ColPali + Groq powered retrieval-augmented generation for PDFs.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check with document count."""
    assert retriever is not None
    return HealthResponse(
        status="ok",
        documents=retriever.count_documents(),
        model=settings.colpali_model_name,
    )


@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)) -> UploadResponse:
    """Upload and ingest a PDF into the vector store."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    # Size check
    contents = await file.read()
    if len(contents) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds {settings.max_upload_mb} MB limit.",
        )

    # Persist a copy
    safe_name = Path(file.filename).name
    dest = settings.upload_dir / safe_name
    dest.write_bytes(contents)

    # Parse
    try:
        page_texts, page_images = parse_pdf(io.BytesIO(contents))
    except PDFParseError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not page_texts:
        raise HTTPException(status_code=400, detail="PDF has no readable pages.")

    # Ingest
    assert retriever is not None
    pages = retriever.add_document(safe_name, page_texts, page_images)

    return UploadResponse(
        filename=safe_name,
        pages=pages,
        message=f"Ingested {pages} pages from {safe_name}.",
    )


@app.post("/ask")
async def ask(request: AskRequest) -> StreamingResponse:
    """Stream a synthesized answer with inline citations.

    Args:
        request: Body containing the question and optional top_k override.
    """
    if llm is None:
        raise HTTPException(
            status_code=503,
            detail="LLM not configured. Set GROQ_API_KEY and restart.",
        )

    assert retriever is not None
    hits = retriever.retrieve(request.question, top_k=request.top_k)
    if not hits:
        raise HTTPException(
            status_code=404, detail="No relevant pages found. Upload papers first."
        )

    context = llm.build_context(hits)

    async def event_stream():
        # Send retrieved sources first as a metadata event
        sources = [
            {"filename": h["filename"], "page": h["page"], "distance": h["distance"]}
            for h in hits
        ]
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"

        # Stream the answer
        async for chunk in llm.stream_answer(request.question, context):
            yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"

        yield "data: {\"type\": \"done\"}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )