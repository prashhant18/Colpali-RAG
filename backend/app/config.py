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
    # Must be a currently served model id (see https://console.groq.com/docs/models).
    # 'llama-3.3-70b-versatile' was decommissioned by Groq (404 on /chat/completions).
    groq_model: str = "openai/gpt-oss-120b"

    # --- ColPali / ColLFM2 ---
    # Lightweight 450M retrieval model (~0.9 GB) that fits in small VRAM.
    colpali_model_name: str = "VAGOsolutions/SauerkrautLM-ColLFM2-450M-v0.1"
    colpali_device: str = "cuda"  # falls back to CPU automatically if unavailable

    # --- Vector DB (ChromaDB) ---
    chroma_persist_dir: Path = Path("./data/chroma")
    collection_name: str = "research_papers"

    # --- Ingestion ---
    upload_dir: Path = Path("./data/uploads")
    max_upload_mb: int = 50
    # Lower DPI = lower memory usage. 96 is a good balance for 8GB-RAM machines.
    render_dpi: int = 96
    # Cap the number of visual tokens per page image (ColLFM2 supports 64-256).
    # Lower = less memory, slightly less detail.
    max_image_tokens: int = 128

    # Run gc.collect() / cuda.empty_cache() only every N embedded pages.
    # Running them after every page is surprisingly expensive (tens of ms
    # each call), which visibly slows down large-document ingestion.
    embed_gc_interval: int = 8

    # Downscale each rendered page so it fits ONE vision tile (~512 px).
    # This disables the processor's multi-tile splitting: measured ~8x
    # faster embedding on a GTX 1650 with negligible loss for paper-level
    # retrieval. Set to 0 to disable downscaling (much slower).
    image_max_side: int = 512

    # --- Retrieval ---
    top_k: int = 5

    # --- Server ---
    host: str = "0.0.0.0"
    port: int = 8000

    # --- CORS ---
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]


settings = Settings()