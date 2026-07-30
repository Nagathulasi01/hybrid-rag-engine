from pydantic import BaseModel, Field
from typing import List, Optional, Any

class QueryRequest(BaseModel):
    query: str = Field(..., description="The user's search query.")
    top_k: int = Field(5, description="Number of context chunks to retrieve.")
    use_hybrid: bool = Field(True, description="Whether to use hybrid search (dense + sparse).")

class Citation(BaseModel):
    chunk_id: str
    content: str
    metadata: dict
    score: float = Field(..., description="Confidence score from the reranker.")

class QueryResponse(BaseModel):
    answer: str = Field(..., description="The LLM-generated answer.")
    citations: List[Citation] = Field(..., description="Supporting context chunks used to generate the answer.")

class IngestResponse(BaseModel):
    status: str
    message: str
    chunks_processed: int = Field(0, description="Number of chunks processed so far.")
    job_id: Optional[str] = Field(None, description="Background ingestion job identifier.")

class IngestJobStatusResponse(BaseModel):
    job_id: str
    status: str
    message: str
    chunks_processed: int = Field(0, description="Number of chunks processed so far.")
    error: Optional[str] = Field(None, description="Error details if ingestion failed.")
