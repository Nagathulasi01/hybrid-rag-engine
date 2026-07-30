import logging
from typing import List
from openai import OpenAI

from ..config import settings

logger = logging.getLogger(__name__)

class Generator:
    """Handles LLM generation using the OpenAI client (which can point to Ollama)."""
    
    def __init__(self):
        api_key = settings.OPENAI_API_KEY or "ollama"
        base_url = settings.LLM_BASE_URL
        self.model_name = settings.LLM_MODEL

        if settings.LLM_PROVIDER == "openai":
            self.model_name = settings.OPENAI_MODEL
            base_url = None

        logger.info(f"Initializing Generator with model: {self.model_name}, Base URL: {base_url}")

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        
    def generate(self, query: str, contexts: List[str]) -> str:
        """
        Generates an answer based on the query and provided context chunks.
        """
        context_str = "\n\n---\n\n".join(contexts)
        
        system_prompt = (
            "You are a helpful, expert AI assistant. Answer the user's question based ONLY "
            "on the provided context. If the context does not contain the answer, say "
            "'I cannot answer this based on the provided context.' Do not hallucinate."
        )
        
        user_prompt = f"Context:\n{context_str}\n\nQuestion:\n{query}"
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=512
        )
        
        return response.choices[0].message.content.strip()
