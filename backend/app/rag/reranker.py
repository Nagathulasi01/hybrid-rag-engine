import logging
from typing import List, Dict, Any
from sentence_transformers import CrossEncoder

from ..config import settings

logger = logging.getLogger(__name__)

class Reranker:
    """Uses a Cross-Encoder to re-rank retrieved chunks based on relevance to the query."""
    
    def __init__(self, model_name: str = settings.RERANKER_MODEL_NAME):
        logger.info(f"Loading Cross-Encoder model: {model_name}")
        self.model = CrossEncoder(model_name, max_length=512)
        
    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Rerank a list of documents for a given query.
        
        Args:
            query (str): The search query.
            documents (List[Dict]): List of dicts, each containing a 'content' key.
            top_k (int): Number of top results to return after reranking.
            
        Returns:
            List[Dict]: Reranked documents with their cross-encoder scores.
        """
        if not documents:
            return []
            
        # Prepare inputs for the cross-encoder: List of (query, document) pairs
        pairs = [[query, doc["content"]] for doc in documents]
        
        # Predict scores
        scores = self.model.predict(pairs)
        
        # Add scores to documents
        for i, doc in enumerate(documents):
            doc["score"] = float(scores[i])
            
        # Sort documents by score in descending order
        reranked_docs = sorted(documents, key=lambda x: x["score"], reverse=True)
        
        return reranked_docs[:top_k]
