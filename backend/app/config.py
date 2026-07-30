import os
from typing import Optional, Tuple


def _load_dotenv() -> None:
    """Load environment variables from a repository-level .env file when present."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
    env_path = os.path.join(repo_root, ".env")

    if not os.path.exists(env_path):
        return

    with open(env_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"\'')
            os.environ.setdefault(key, value)


_load_dotenv()


class Settings:
    """Centralized application configuration with startup validation."""

    def __init__(self) -> None:
        self.APP_TITLE = os.getenv("APP_TITLE", "Hybrid RAG API")
        self.APP_DESCRIPTION = os.getenv(
            "APP_DESCRIPTION",
            "Portfolio-grade Hybrid RAG Pipeline API",
        )
        self.APP_VERSION = os.getenv("APP_VERSION", "1.0.0")
        self.APP_ENV = os.getenv("APP_ENV", "development")

        self.QDRANT_URL = os.getenv("QDRANT_URL") or (
            ":memory:" if self.APP_ENV != "production" else None
        )
        self.COLLECTION_NAME = os.getenv("COLLECTION_NAME", "hybrid_rag_docs")

        self.LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
        self.LLM_MODEL = os.getenv("LLM_MODEL", "llama3")
        self.OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

        self.DENSE_MODEL_NAME = os.getenv("DENSE_MODEL_NAME", "all-MiniLM-L6-v2")
        self.SPARSE_MODEL_NAME = os.getenv(
            "SPARSE_MODEL_NAME",
            "prithivida/Splade_PP_en_v1",
        )
        self.RERANKER_MODEL_NAME = os.getenv(
            "RERANKER_MODEL_NAME",
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
        )

        self.TEMP_UPLOAD_DIR = os.getenv("TEMP_UPLOAD_DIR", "temp_uploads")
        self.CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
        self.CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
        self.RRF_K = int(os.getenv("RRF_K", "60"))
        self.INITIAL_RETRIEVAL_MULTIPLIER = int(
            os.getenv("INITIAL_RETRIEVAL_MULTIPLIER", "4")
        )
        self.DENSE_VECTOR_SIZE = int(os.getenv("DENSE_VECTOR_SIZE", "384"))
        self.VECTOR_DISTANCE = os.getenv("VECTOR_DISTANCE", "COSINE")
        self.ALLOWED_EXTENSIONS: Tuple[str, ...] = tuple(
            ext.strip()
            for ext in os.getenv("ALLOWED_EXTENSIONS", ".pdf,.txt").split(",")
            if ext.strip()
        )

        self.validate()

    def validate(self) -> None:
        errors: list[str] = []

        if self.LLM_PROVIDER not in {"openai", "ollama"}:
            errors.append("LLM_PROVIDER must be either 'openai' or 'ollama'.")

        if self.LLM_PROVIDER == "openai" and not self.OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY is required when LLM_PROVIDER=openai.")

        if not self.LLM_BASE_URL:
            errors.append("LLM_BASE_URL cannot be empty.")

        if self.APP_ENV == "production" and not self.QDRANT_URL:
            errors.append("QDRANT_URL is required when APP_ENV=production.")

        if self.CHUNK_SIZE <= 0:
            errors.append("CHUNK_SIZE must be greater than zero.")

        if self.CHUNK_OVERLAP < 0:
            errors.append("CHUNK_OVERLAP cannot be negative.")

        if self.RRF_K <= 0:
            errors.append("RRF_K must be greater than zero.")

        if self.INITIAL_RETRIEVAL_MULTIPLIER <= 0:
            errors.append("INITIAL_RETRIEVAL_MULTIPLIER must be greater than zero.")

        if self.DENSE_VECTOR_SIZE <= 0:
            errors.append("DENSE_VECTOR_SIZE must be greater than zero.")

        if not self.ALLOWED_EXTENSIONS:
            errors.append("ALLOWED_EXTENSIONS must include at least one supported extension.")

        if errors:
            raise RuntimeError("Invalid configuration: " + " ".join(errors))


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


settings = get_settings()
