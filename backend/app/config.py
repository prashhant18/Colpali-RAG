"""Application configuration using pydantic-settings."""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for the RAG application.

    Reads from environment variables or a `.env` file in the backend directory.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Groq LLM ---
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    # --- ColPali ---
    colpali_model_name: str = "vidore/colpali-v1.2"
    colpali_device: str = "cpu"  # or "cuda" if GPU available

    # --- Vector DB (ChromaDB) ---
    chroma_persist_dir: Path = Path("./data/chroma")
    collection_name: str = "research_papers"

    # --- Ingestion ---
    upload_dir: Path = Path("./data/uploads")
    max_upload_mb: int = 50

    # --- Retrieval ---
    top_k: int = 5

    # --- Server ---
    host: str = "0.0.0.0"
    port: int = 8000

    # --- CORS ---
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]


settings = Settings()