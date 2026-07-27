import os
import logging
from typing import List
from openai import OpenAI

logger = logging.getLogger(__name__)

class Generator:
    """Handles LLM generation using the OpenAI client (which can point to Ollama)."""
    
    def __init__(self):
        # Default to Ollama local endpoint if OPENAI_API_KEY is not set
        api_key = os.getenv("OPENAI_API_KEY", "ollama")
        base_url = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
        self.model_name = os.getenv("LLM_MODEL", "llama3") # Default to Llama 3 for local
        
        # If the user explicitly provided an OpenAI key and no base URL, let it default to OpenAI's actual URL
        if api_key != "ollama" and os.getenv("LLM_BASE_URL") is None:
            base_url = None
            self.model_name = os.getenv("LLM_MODEL", "gpt-4o-mini")
            
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
