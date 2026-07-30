import logging
from typing import List, Tuple, Dict, Any
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding

from ..config import settings

logger = logging.getLogger(__name__)

class Embedder:
    """Handles generation of dense and sparse embeddings for Hybrid RAG."""
    
    def __init__(self, dense_model_name: str = settings.DENSE_MODEL_NAME, sparse_model_name: str = settings.SPARSE_MODEL_NAME):
        # Dense Model (Sentence-Transformers)
        logger.info(f"Loading dense model: {dense_model_name}")
        self.dense_model = SentenceTransformer(dense_model_name)
        
        # Sparse Model (FastEmbed)
        logger.info(f"Loading sparse model: {sparse_model_name}")
        # Using SPLADE or BM25 from FastEmbed. SPLADE is excellent for sparse vectors.
        self.sparse_model = SparseTextEmbedding(model_name=sparse_model_name)
        
    def embed_dense(self, texts: List[str]) -> List[List[float]]:
        """Generate dense embeddings."""
        embeddings = self.dense_model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
        
    def embed_sparse(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Generate sparse embeddings. 
        Returns list of dicts with 'indices' and 'values' for Qdrant.
        """
        sparse_embeddings = list(self.sparse_model.embed(texts))
        
        # Convert fastembed output (SparseEmbedding objects) to Qdrant compatible format
        qdrant_sparse = []
        for emb in sparse_embeddings:
            qdrant_sparse.append({
                "indices": emb.indices.tolist(),
                "values": emb.values.tolist()
            })
        return qdrant_sparse

    def embed_queries(self, query: str) -> Tuple[List[float], Dict[str, Any]]:
        """Embed a single query into both dense and sparse representations."""
        dense = self.embed_dense([query])[0]
        sparse = self.embed_sparse([query])[0]
        return dense, sparse
