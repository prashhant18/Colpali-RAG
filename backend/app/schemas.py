"""Pydantic schemas for API request/response models."""

from typing import Optional
from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    """Response after uploading and ingesting a PDF."""

    filename: str
    pages: int
    status: str = "ingested"
    message: str


class AskRequest(BaseModel):
    """Request body for the `/ask` chat endpoint."""

    question: str = Field(..., min_length=1, description="Natural-language question")
    top_k: Optional[int] = Field(
        default=None, ge=1, le=20, description="Override retrieval top-K"
    )


class Citation(BaseModel):
    """A single source citation."""

    filename: str
    page: int
    text: str


class AskResponse(BaseModel):
    """Final (non-streamed) response for the `/ask` endpoint."""

    answer: str
    citations: list[Citation]


class HealthResponse(BaseModel):
    """Health check payload."""

    status: str
    documents: int
    model: str