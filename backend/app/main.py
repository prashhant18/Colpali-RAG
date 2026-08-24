"""FastAPI application entrypoint for the RAG research-paper assistant."""

import json
import logging
import queue
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from .config import settings
from .llm import GroqLLM
from .pdf_parser import PDFParseError, iter_pdf_pages
from .retriever import ColPaliRetriever
from .schemas import AskRequest, HealthResponse, UploadResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global singletons (initialized lazily to keep startup fast)
# ---------------------------------------------------------------------------
retriever: ColPaliRetriever | None = None
llm: GroqLLM | None = None

# Only one ingestion at a time: embedding is memory-heavy and this app targets
# an 8 GB RAM machine — concurrent uploads would thrash memory.
ingest_semaphore = threading.Semaphore(1)


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
    description="SauerkrautLM-ColLFM2 + Groq powered retrieval-augmented generation for PDFs.",
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


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)) -> StreamingResponse:
    """Upload and ingest a PDF into the vector store.

    Progress is streamed back as server-sent events so the frontend can show
    a live progress bar. The heavy work (rendering + embedding) runs in a
    worker thread, keeping the event loop free so other requests stay
    responsive during ingestion.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    safe_name = Path(file.filename).name
    dest = settings.upload_dir / safe_name
    max_bytes = settings.max_upload_mb * 1024 * 1024

    if not ingest_semaphore.acquire(blocking=False):
        raise HTTPException(
            status_code=409,
            detail="Another upload is currently being processed. Please wait.",
        )

    # Stream the upload to disk in fixed-size chunks instead of holding the
    # whole file in memory; enforce the size limit while copying.
    try:
        bytes_written = 0
        with dest.open("wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                bytes_written += len(chunk)
                if bytes_written > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds {settings.max_upload_mb} MB limit.",
                    )
                out.write(chunk)
    except HTTPException:
        dest.unlink(missing_ok=True)
        ingest_semaphore.release()
        raise

    progress_queue: queue.Queue = queue.Queue()
    cancel_event = threading.Event()

    def ingest_worker() -> None:
        """Run parsing + embedding off the event loop; report via the queue."""
        success = False
        try:
            def on_progress(current: int, total: int) -> None:
                progress_queue.put(
                    {"type": "progress", "current": current, "total": total}
                )

            # Wrap the page iterator so a client disconnect aborts ingestion
            # after the current page instead of embedding the whole document.
            def pages():
                for item in iter_pdf_pages(dest, dpi=settings.render_dpi):
                    if cancel_event.is_set():
                        return
                    yield item

            assert retriever is not None
            count = retriever.ingest_document(
                safe_name, pages(), progress_cb=on_progress
            )

            if cancel_event.is_set():
                return

            result = UploadResponse(
                filename=safe_name,
                pages=count,
                message=f"Ingested {count} pages from {safe_name}.",
            )
            progress_queue.put({"type": "done", **result.model_dump()})
            success = True
        except PDFParseError as exc:
            progress_queue.put({"type": "error", "detail": str(exc)})
        except Exception as exc:  # noqa: BLE001
            logger.exception("Ingestion failed for %s", safe_name)
            progress_queue.put({"type": "error", "detail": f"Ingestion failed: {exc}"})
        finally:
            progress_queue.put(None)  # sentinel: stream ends
            ingest_semaphore.release()
            if not success:
                dest.unlink(missing_ok=True)

    async def event_stream():
        yield f"data: {json.dumps({'type': 'start', 'filename': safe_name})}\n\n"
        yield f"data: {json.dumps({'type': 'parsing'})}\n\n"

        worker = threading.Thread(target=ingest_worker, daemon=True)
        worker.start()
        try:
            while True:
                # Blocking get() runs in the threadpool so the event loop (and
                # therefore /ask, /health, ...) stays responsive throughout.
                event = await run_in_threadpool(progress_queue.get)
                if event is None:
                    break
                yield f"data: {json.dumps(event)}\n\n"
        finally:
            # Client disconnected mid-ingest: tell the worker to stop early.
            cancel_event.set()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
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
    # Retrieval embeds the query with the local model; run it in a worker
    # thread so model inference never blocks the event loop either.
    hits = await run_in_threadpool(retriever.retrieve, request.question, request.top_k)
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

        # Stream the answer. Any failure (bad model id, network, quota, ...)
        # is reported as an SSE 'error' event so the UI can show it instead
        # of the whole request crashing mid-stream.
        try:
            async for chunk in llm.stream_answer(request.question, context):
                yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"
        except Exception as exc:
            logger.exception("LLM streaming failed for question: %s", request.question)
            detail = str(exc) or exc.__class__.__name__
            yield f"data: {json.dumps({'type': 'error', 'detail': detail})}\n\n"
            return

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