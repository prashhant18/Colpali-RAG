"""Groq LLM wrapper with streaming support."""

import json
import logging
from typing import AsyncIterator

import httpx

from .config import settings

logger = logging.getLogger(__name__)

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

SYSTEM_PROMPT = """You are a research assistant helping users understand academic papers.

You will be given retrieved page excerpts from research papers, each tagged with
a source filename and page number. Answer the user's question using ONLY the
provided context. If the context does not contain the answer, say so clearly.

CRITICAL CITATION RULES:
- Every factual claim you make must be followed by an inline citation in the
  format [Author et al., YEAR, p. N] where YEAR is the publication year and N
  is the page number from the source metadata.
- If the year is unknown, use [Source, p. N].
- Group citations when multiple pages support the same claim, e.g.
  [Author et al., 2021, p. 3; Author et al., 2020, p. 7].
- Be concise. Use short paragraphs and bullet points where helpful.
"""


class GroqLLM:
    """Thin async client for Groq's chat completions API."""

    def __init__(self) -> None:
        if not settings.groq_api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Add it to backend/.env or environment."
            )
        self._api_key = settings.groq_api_key
        self._model = settings.groq_model

    def build_context(self, hits: list[dict]) -> str:
        """Build a context block from retrieved page hits."""
        blocks: list[str] = []
        for i, hit in enumerate(hits, start=1):
            blocks.append(
                f"[Source {i}] Filename: {hit['filename']} | Page: {hit['page']}\n"
                f"{hit['text']}\n"
            )
        return "\n\n".join(blocks)

    async def stream_answer(
        self, question: str, context: str
    ) -> AsyncIterator[str]:
        """Stream the LLM answer token-by-token.

        Args:
            question: The user's question.
            context: The retrieved context block.

        Yields:
            Text chunks of the generated answer.
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"CONTEXT:\n{context}\n\n"
                    f"QUESTION: {question}\n\n"
                    "Answer the question using the context above, with inline "
                    "citations in the format [Author et al., YEAR, p. N]."
                ),
            },
        ]

        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 1024,
            "stream": True,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST", GROQ_API_URL, json=payload, headers=headers
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:") :].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        delta = chunk["choices"][0]["delta"].get("content", "")
                        if delta:
                            yield delta
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue