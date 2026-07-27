import os
import shutil
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from contextlib import asynccontextmanager

from .schemas import QueryRequest, QueryResponse, IngestResponse
from .rag.document_parser import parse_and_chunk
from .rag.retriever import RAGRetriever
from .rag.generator import Generator

logger = logging.getLogger(__name__)

# Initialize components globally to avoid reloading on every request
retriever = None
generator = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever, generator
    logger.info("Initializing RAG components on startup...")
    # Load models and connect to DB on startup
    retriever = RAGRetriever()
    generator = Generator()
    logger.info("RAG components initialized.")
    yield
    # Cleanup on shutdown if necessary
    logger.info("Shutting down API...")

app = FastAPI(
    title="Hybrid RAG API",
    description="Portfolio-grade Hybrid RAG Pipeline API",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "API is running."}

@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(file: UploadFile = File(...)):
    if not file.filename.endswith(('.pdf', '.txt')):
        logger.warning(f"Invalid file type uploaded: {file.filename}")
        raise HTTPException(status_code=400, detail="Only .pdf and .txt files are supported.")
    
    # Save file temporarily in a local temp directory
    temp_dir = os.path.join(os.getcwd(), "temp_uploads")
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, file.filename)
    
    try:
        logger.info(f"Receiving file: {file.filename}")
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 1. Parse and chunk
        chunks = parse_and_chunk(temp_file_path)
        
        # 2. Ingest into Qdrant via Retriever
        retriever.ingest(chunks)
        
        logger.info(f"Successfully ingested {len(chunks)} chunks from {file.filename}")
        return IngestResponse(
            status="success",
            message=f"Successfully ingested {file.filename}",
            chunks_processed=len(chunks)
        )
    except Exception as e:
        logger.error(f"Error ingesting file {file.filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            logger.info(f"Cleaned up temporary file: {temp_file_path}")

@app.post("/query", response_model=QueryResponse)
async def query_pipeline(request: QueryRequest):
    try:
        # 1. Retrieve & Rerank Context
        top_k_results = retriever.search(
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
        answer = generator.generate(request.query, context_texts)
        
        # 3. Format Response
        logger.info(f"Generated answer for query: '{request.query}'")
        return QueryResponse(
            answer=answer,
            citations=top_k_results
        )
    except Exception as e:
        logger.error(f"Error processing query '{request.query}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
