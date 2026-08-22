"""ColPali-based multimodal retriever.

Treats each PDF page as an image and uses ColPali's late-interaction
mechanism to retrieve the most relevant pages for a text query.
"""

import logging
from typing import Any

import chromadb
import torch
from chromadb.config import Settings as ChromaSettings
from colpali_engine.models import ColPali, ColPaliProcessor  # type: ignore
from PIL import Image

from .config import settings

logger = logging.getLogger(__name__)


class ColPaliRetriever:
    """Wraps the ColPali model, ChromaDB store, and retrieval logic."""

    def __init__(self) -> None:
        self._model: ColPali | None = None
        self._processor: ColPaliProcessor | None = None
        self._device = settings.colpali_device

        # ChromaDB persistent client
        settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(settings.chroma_persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=settings.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # Model loading (lazy, on first use)
    # ------------------------------------------------------------------
    def _ensure_model(self) -> None:
        """Load the ColPali model and processor on first use."""
        if self._model is not None and self._processor is not None:
            return

        logger.info("Loading ColPali model: %s", settings.colpali_model_name)
        self._model = ColPali.from_pretrained(
            settings.colpali_model_name,
            torch_dtype=torch.bfloat16 if self._device == "cuda" else torch.float32,
            device_map=self._device,
        ).eval()
        self._processor = ColPaliProcessor.from_pretrained(settings.colpali_model_name)

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------
    def _embed_images(self, images: list[Image.Image]) -> list[list[float]]:
        """Compute ColPali embeddings for a list of page images."""
        self._ensure_model()
        assert self._model is not None and self._processor is not None

        batch = self._processor.process_images(images)
        with torch.no_grad():
            embeddings = self._model(**batch.to(self._device))
        # embeddings: (batch, num_patches, dim) -> mean-pool to a single vector
        pooled = embeddings.mean(dim=1).cpu().tolist()
        return [list(map(float, vec)) for vec in pooled]

    def _embed_query(self, query: str) -> list[float]:
        """Compute a ColPali embedding for a text query."""
        self._ensure_model()
        assert self._model is not None and self._processor is not None

        batch = self._processor.process_queries([query])
        with torch.no_grad():
            embedding = self._model(**batch.to(self._device))
        pooled = embedding.mean(dim=1).cpu().tolist()[0]
        return list(map(float, pooled))

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------
    def add_document(
        self,
        filename: str,
        page_texts: list[str],
        page_images: list[Image.Image],
    ) -> int:
        """Embed and store all pages of a document.

        Args:
            filename: Source PDF filename.
            page_texts: Per-page extracted text.
            page_images: Per-page rendered images.

        Returns:
            Number of pages ingested.
        """
        embeddings = self._embed_images(page_images)

        ids: list[str] = []
        metadatas: list[dict[str, Any]] = []
        documents: list[str] = []

        for i, (text, emb) in enumerate(zip(page_texts, embeddings)):
            page_num = i + 1
            doc_id = f"{filename}::page::{page_num}"
            ids.append(doc_id)
            metadatas.append(
                {
                    "filename": filename,
                    "page": page_num,
                    "total_pages": len(page_texts),
                }
            )
            documents.append(text)

        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        logger.info("Ingested %d pages from %s", len(page_texts), filename)
        return len(page_texts)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def retrieve(
        self, query: str, top_k: int | None = None
    ) -> list[dict[str, Any]]:
        """Retrieve the top-K most relevant pages for a query.

        Args:
            query: Natural-language question.
            top_k: Number of results (defaults to settings.top_k).

        Returns:
            A list of dicts with keys: filename, page, text, distance.
        """
        k = top_k or settings.top_k
        query_emb = self._embed_query(query)

        results = self._collection.query(
            query_embeddings=[query_emb],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        hits: list[dict[str, Any]] = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        for doc, meta, dist in zip(docs, metas, dists):
            hits.append(
                {
                    "filename": meta["filename"],
                    "page": int(meta["page"]),
                    "text": doc,
                    "distance": float(dist),
                }
            )
        return hits

    def count_documents(self) -> int:
        """Return the number of stored page chunks."""
        return self._collection.count()