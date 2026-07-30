import asyncio
import os
import shutil
import logging
import uuid
from typing import Any, Dict
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from .config import settings
from .schemas import QueryRequest, QueryResponse, IngestResponse, IngestJobStatusResponse
from .rag.document_parser import parse_and_chunk
from .rag.retriever import RAGRetriever
from .rag.generator import Generator

logger = logging.getLogger(__name__)

# Initialize components globally to avoid reloading on every request
retriever = None
generator = None
ingest_jobs: Dict[str, Dict[str, Any]] = {}


def _ensure_retriever():
    global retriever
    if retriever is None:
        retriever = RAGRetriever()
    return retriever


def _ensure_generator():
    global generator
    if generator is None:
        generator = Generator()
    return generator


def _update_job(job_id: str, **updates: Any) -> None:
    if job_id not in ingest_jobs:
        ingest_jobs[job_id] = {}
    ingest_jobs[job_id].update(updates)


async def _process_ingestion_job(job_id: str, temp_file_path: str, filename: str) -> None:
    _update_job(job_id, status="processing", message=f"Processing {filename}", chunks_processed=0, error=None)
    try:
        retriever_instance = _ensure_retriever()
        chunks = await asyncio.to_thread(parse_and_chunk, temp_file_path)
        await asyncio.to_thread(retriever_instance.ingest, chunks)
        _update_job(
            job_id,
            status="completed",
            message=f"Successfully ingested {filename}",
            chunks_processed=len(chunks),
            error=None,
        )
    except Exception as exc:
        logger.error(f"Background ingestion failed for {filename}: {exc}", exc_info=True)
        _update_job(
            job_id,
            status="failed",
            message=f"Ingestion failed for {filename}",
            chunks_processed=0,
            error=str(exc),
        )
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            logger.info(f"Cleaned up temporary file: {temp_file_path}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever, generator
    logger.info("Initializing RAG components on startup...")
    try:
        # Load models and connect to DB on startup
        retriever = RAGRetriever()
        generator = Generator()
        logger.info("RAG components initialized.")
    except Exception as exc:
        logger.warning("RAG component initialization failed during startup: %s", exc, exc_info=True)
        retriever = None
        generator = None
    yield
    # Cleanup on shutdown if necessary
    logger.info("Shutting down API...")

app = FastAPI(
    title=settings.APP_TITLE,
    description=settings.APP_DESCRIPTION,
    version=settings.APP_VERSION,
    lifespan=lifespan
)


def _get_qdrant_status() -> dict:
    if retriever is None or getattr(retriever, "client", None) is None:
        return {"status": "not_initialized", "detail": "Qdrant client has not been initialized yet."}

    qdrant_url = settings.QDRANT_URL or ":memory:"
    if qdrant_url == ":memory:":
        return {"status": "ok", "detail": "Using in-memory Qdrant storage."}

    try:
        retriever.client.get_collections()
        return {"status": "ok", "detail": f"Connected to {qdrant_url}."}
    except Exception as exc:
        logger.warning(f"Qdrant health check failed: {exc}")
        return {"status": "error", "detail": str(exc)}


def _get_llm_status() -> dict:
    if settings.LLM_PROVIDER == "openai":
        if settings.OPENAI_API_KEY:
            return {"status": "ok", "detail": "OpenAI provider configured with an API key."}
        return {"status": "error", "detail": "OpenAI provider selected but OPENAI_API_KEY is missing."}

    if settings.LLM_BASE_URL:
        return {"status": "ok", "detail": f"Ollama provider configured with {settings.LLM_BASE_URL}."}

    return {"status": "error", "detail": "Ollama provider selected but LLM_BASE_URL is missing."}


@app.get("/health")
async def health_check():
    qdrant_status = _get_qdrant_status()
    llm_status = _get_llm_status()
    overall_status = "ok"
    if qdrant_status["status"] != "ok" or llm_status["status"] != "ok":
        overall_status = "degraded"

    return {
        "status": overall_status,
        "message": "API is running.",
        "dependencies": {
            "qdrant": qdrant_status,
            "llm": llm_status,
        },
    }


@app.get("/ready")
async def readiness_check():
    qdrant_status = _get_qdrant_status()
    llm_status = _get_llm_status()
    is_ready = qdrant_status["status"] == "ok" and llm_status["status"] == "ok"

    payload = {
        "status": "ready" if is_ready else "not_ready",
        "dependencies": {
            "qdrant": qdrant_status,
            "llm": llm_status,
        },
    }
    status_code = 200 if is_ready else 503
    return JSONResponse(status_code=status_code, content=payload)

@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    if not file.filename.endswith(settings.ALLOWED_EXTENSIONS):
        logger.warning(f"Invalid file type uploaded: {file.filename}")
        raise HTTPException(status_code=400, detail=f"Only {', '.join(settings.ALLOWED_EXTENSIONS)} files are supported.")

    # Save file temporarily in a local temp directory
    temp_dir = os.path.join(os.getcwd(), settings.TEMP_UPLOAD_DIR)
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, file.filename)

    job_id = str(uuid.uuid4())
    ingest_jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "message": f"Queued ingestion for {file.filename}",
        "chunks_processed": 0,
        "error": None,
    }

    try:
        logger.info(f"Receiving file: {file.filename}")
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        background_tasks.add_task(_process_ingestion_job, job_id, temp_file_path, file.filename)
        return IngestResponse(
            status="queued",
            message=f"Ingestion started for {file.filename}",
            chunks_processed=0,
            job_id=job_id,
        )
    except Exception as e:
        logger.error(f"Error ingesting file {file.filename}: {e}", exc_info=True)
        _update_job(job_id, status="failed", message=f"Ingestion failed for {file.filename}", chunks_processed=0, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ingest/{job_id}/status", response_model=IngestJobStatusResponse)
async def ingest_status(job_id: str):
    job = ingest_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return IngestJobStatusResponse(**job)

@app.post("/query", response_model=QueryResponse)
async def query_pipeline(request: QueryRequest):
    try:
        retriever_instance = _ensure_retriever()
        generator_instance = _ensure_generator()

        # 1. Retrieve & Rerank Context
        top_k_results = retriever_instance.search(
            query=request.query,
            top_k=request.top_k,
            use_hybrid=request.use_hybrid
        )
        
        if not top_k_results:
            return QueryResponse(
                answer="I could not find any relevant information to answer your query.",
                citations=[]
            )
            
        # 2. Generate Answer
        context_texts = [res.content for res in top_k_results]
        answer = generator_instance.generate(request.query, context_texts)
        
        # 3. Format Response
        logger.info(f"Generated answer for query: '{request.query}'")
        return QueryResponse(
            answer=answer,
            citations=top_k_results
        )
    except Exception as e:
        logger.error(f"Error processing query '{request.query}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
