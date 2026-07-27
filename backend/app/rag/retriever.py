import os
import uuid
import logging
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, SparseVectorParams, PointStruct

logger = logging.getLogger(__name__)

from .embedder import Embedder
from .reranker import Reranker

class RAGRetriever:
    """Handles communication with Qdrant and executes Hybrid Search + RRF."""
    
    def __init__(self, collection_name: str = "hybrid_rag_docs"):
        self.collection_name = collection_name
        
        # Connect to Qdrant (use local in-memory or Docker container if QDRANT_URL is set)
        qdrant_url = os.getenv("QDRANT_URL", ":memory:")
        logger.info(f"Connecting to Qdrant at {qdrant_url}")
        
        if qdrant_url == ":memory:":
            self.client = QdrantClient(location=":memory:")
        else:
            self.client = QdrantClient(url=qdrant_url)
            
        self.embedder = Embedder()
        self.reranker = Reranker()
        
        self._ensure_collection()
        
    def _ensure_collection(self):
        """Creates the Qdrant collection if it doesn't exist, configuring both dense and sparse vectors."""
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": VectorParams(
                        size=384, # all-MiniLM-L6-v2 dimension
                        distance=Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(
                        modifier=None
                    )
                }
            )
            logger.info(f"Collection '{self.collection_name}' created.")

    def ingest(self, chunks: List[Dict[str, str]]):
        """Embeds and uploads chunks to Qdrant."""
        if not chunks:
            return
            
        texts = [chunk["content"] for chunk in chunks]
        
        # Generate embeddings
        dense_embs = self.embedder.embed_dense(texts)
        sparse_embs = self.embedder.embed_sparse(texts)
        
        points = []
        for i, chunk in enumerate(chunks):
            point_id = str(uuid.uuid4())
            points.append(
                PointStruct(
                    id=point_id,
                    vector={
                        "dense": dense_embs[i],
                        "sparse": sparse_embs[i]
                    },
                    payload={
                        "content": chunk["content"],
                        "metadata": chunk.get("metadata", {})
                    }
                )
            )
            
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        logger.info(f"Ingested {len(points)} chunks into Qdrant.")

    def _rrf(self, dense_results: List[Any], sparse_results: List[Any], k: int = 60) -> List[Dict[str, Any]]:
        """
        Reciprocal Rank Fusion (RRF) algorithm.
        Score = 1 / (k + rank)
        """
        rrf_scores = {}
        docs = {}
        
        # Process dense results
        for rank, res in enumerate(dense_results):
            doc_id = res.id
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0
                docs[doc_id] = res
            rrf_scores[doc_id] += 1.0 / (k + rank + 1)
            
        # Process sparse results
        for rank, res in enumerate(sparse_results):
            doc_id = res.id
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0
                docs[doc_id] = res
            rrf_scores[doc_id] += 1.0 / (k + rank + 1)
            
        # Sort by RRF score
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Format for reranker
        formatted_results = []
        for doc_id, score in sorted_docs:
            formatted_results.append({
                "chunk_id": str(doc_id),
                "content": docs[doc_id].payload["content"],
                "metadata": docs[doc_id].payload.get("metadata", {}),
                "rrf_score": score
            })
            
        return formatted_results

    def search(self, query: str, top_k: int = 5, use_hybrid: bool = True):
        """Performs search (Hybrid or Dense only) and applies Cross-Encoder reranking."""
        
        dense_q, sparse_q = self.embedder.embed_queries(query)
        
        # Increase initial retrieval limit for better reranking pool
        initial_k = top_k * 4 
        
        dense_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=("dense", dense_q),
            limit=initial_k
        )
        
        if use_hybrid:
            sparse_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=("sparse", sparse_q),
                limit=initial_k
            )
            
            # Combine via RRF
            fused_results = self._rrf(dense_results, sparse_results)
        else:
            # Just format dense results
            fused_results = []
            for res in dense_results:
                fused_results.append({
                    "chunk_id": str(res.id),
                    "content": res.payload["content"],
                    "metadata": res.payload.get("metadata", {}),
                    "score": res.score
                })
                
        # Rerank with Cross-Encoder
        reranked_results = self.reranker.rerank(query, fused_results, top_k=top_k)
        
        # Format final output matching the Citation schema
        from ..schemas import Citation
        
        final_citations = []
        for res in reranked_results:
            final_citations.append(
                Citation(
                    chunk_id=res["chunk_id"],
                    content=res["content"],
                    metadata=res["metadata"],
                    score=res["score"]
                )
            )
            
        return final_citations
