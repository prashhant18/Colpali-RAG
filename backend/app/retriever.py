"""Multimodal retriever using the SauerkrautLM-ColLFM2-450M model.

SauerkrautLM-ColLFM2-450M is a lightweight (450M param, ~0.9 GB VRAM in
bfloat16) vision-language retrieval model. Like ColPali, each PDF page is
treated as an image, and late-interaction multi-vector embeddings are used to
retrieve the most relevant pages for a text query.

This is a drop-in replacement for the 5.85 GB `vidore/colpali-v1.2` model:
  - Much smaller: fits in 4-6 GB VRAM.
  - Still captures tables, figures, and complex layouts via page images.

The model requires VAGO's `sauerkrautlm-colpali` package (see README):
    pip install git+https://github.com/VAGOsolutions/sauerkrautlm-colpali

Memory notes (important for low-RAM machines):
  - The `sauerkrautlm-colpali` package's `from_pretrained` drops the
    `device_map` argument, so the base model always loads on CPU. We therefore
    explicitly move it to the GPU after loading (`.to(device)`).
  - Pages are embedded ONE AT A TIME (not batched) to keep peak memory low,
    and we free GPU/CPU memory between pages.
"""

import gc
import logging
import os
import time
from collections.abc import Callable, Iterator
from typing import Any

import chromadb
import torch
from chromadb.config import Settings as ChromaSettings
from PIL import Image

from .config import settings

logger = logging.getLogger(__name__)

# The `sauerkrautlm_colpali` package provides the ColLFM2 architecture for the
# SauerkrautLM-ColLFM2-450M model. Import lazily so the app can still start
# with a clear error message if the package is missing.
try:
    from sauerkrautlm_colpali.models import ColLFM2, ColLFM2Processor

    _MODEL_CLASS = ColLFM2
    _PROCESSOR_CLASS = ColLFM2Processor
except ImportError:  # pragma: no cover - handled at runtime
    _MODEL_CLASS = None
    _PROCESSOR_CLASS = None


class ColPaliRetriever:
    """Wraps the ColLFM2 (SauerkrautLM-ColLFM2-450M) model, ChromaDB, retrieval."""

    def __init__(self) -> None:
        self._model = None
        self._processor = None
        self._device = self._resolve_device(settings.colpali_device)

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
    # Device helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_device(requested: str) -> str:
        """Resolve the requested device to an actually usable one.

        If the user requests CUDA but torch was built without CUDA (or no GPU
        is present), fall back to CPU so the app still runs instead of raising
        "torch not compiled with CUDA enabled".
        """
        if requested == "cuda":
            if torch.cuda.is_available():
                logger.info("CUDA available; using GPU.")
                return "cuda"
            logger.warning(
                "CUDA was requested but torch is not compiled with CUDA "
                "(or no GPU is available). Falling back to CPU."
            )
        return "cpu"

    @staticmethod
    def _free_memory() -> None:
        """Release GPU and CPU memory between page embeddings."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Model loading (lazy, on first use)
    # ------------------------------------------------------------------
    def _ensure_model(self) -> None:
        """Load the ColLFM2 model and processor on first use.

        Performance notes (GTX 1650-class GPUs, 8 GB RAM machines):
          - The sauerkrautlm-colpali package's ``from_pretrained`` ignores
            ``torch_dtype``/``device_map`` kwargs and always builds an fp32
            model on CPU, so we move/cast it ourselves afterwards.
          - fp16 is used on CUDA (fast path on Turing and newer); CPU keeps
            fp32. A warmup forward pass runs once so the first real page
            isn't slowed by CUDA kernel initialisation.
        """
        if self._model is not None and self._processor is not None:
            return

        if _MODEL_CLASS is None or _PROCESSOR_CLASS is None:
            raise RuntimeError(
                "The 'sauerkrautlm-colpali' package is not installed. "
                "Install it with:\n"
                "  pip install git+https://github.com/VAGOsolutions/sauerkrautlm-colpali"
            )

        started = time.perf_counter()
        logger.info(
            "Loading model: %s (device=%s)", settings.colpali_model_name, self._device
        )
        # NOTE: intentionally no torch_dtype/device_map kwargs — the custom
        # from_pretrained in the package ignores them.
        self._model = _MODEL_CLASS.from_pretrained(settings.colpali_model_name)

        # Repair vision-tower weights that the package loader fails to map
        # (checkpoint stores them under 'vision_tower.vision_model.*').
        self._repair_vision_tower_weights(self._model)

        dtype = torch.float16 if self._device == "cuda" else torch.float32
        try:
            self._model = self._model.to(device=self._device, dtype=dtype)
        except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
            logger.warning(
                "Could not place model on %s (%s). Falling back to CPU fp32.",
                self._device,
                exc,
            )
            self._device = "cpu"
            self._model = self._model.to("cpu")
        self._model.eval()

        self._processor = _PROCESSOR_CLASS.from_pretrained(
            settings.colpali_model_name,
            max_image_tokens=settings.max_image_tokens,
            min_image_tokens=64,
        )

        # Single-tile mode: the LFM2-VL processor otherwise splits each page
        # into up to ten 512px tiles (+ thumbnail), which multiplies vision
        # compute by ~8x while barely helping page-level retrieval. Combined
        # with `_prepare_page`'s downscale this keeps one page = one tile.
        img_proc = getattr(self._processor.processor, "image_processor", None)
        if img_proc is not None and hasattr(img_proc, "do_image_splitting"):
            img_proc.do_image_splitting = False
            logger.info("Vision tiling disabled (single-tile fast embedding).")

        logger.info(
            "Model ready on %s (%s) in %.1fs.",
            self._device.upper(),
            next(self._model.parameters()).dtype,
            time.perf_counter() - started,
        )
        self._warmup()

    # ------------------------------------------------------------------
    # Weight repair / warmup helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _open_safetensors() -> dict | None:
        """Return the model's safetensors state dict from the HF cache."""
        try:
            from huggingface_hub import snapshot_download
            from safetensors.torch import load_file

            local_dir = snapshot_download(
                settings.colpali_model_name,
                allow_patterns=["model.safetensors"],
                local_files_only=True,
            )
            return load_file(os.path.join(local_dir, "model.safetensors"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not open cached safetensors locally: %s", exc)
            return None

    @classmethod
    def _repair_vision_tower_weights(cls, model: Any) -> None:
        """Fix the package's checkpoint-key mismatch for the vision tower.

        The package maps 'model.*' -> 'base_model.model.*' but the checkpoint
        stores vision weights one level deeper ('vision_tower.vision_model.*')
        than the module tree expects ('vision_tower.*'). Those keys end up
        'unexpected' and the vision encoder silently keeps RANDOM weights,
        which wrecks retrieval quality for every stored embedding.
        """
        state_dict = cls._open_safetensors()
        if state_dict is None:
            logger.warning(
                "Vision-weight repair skipped: safetensors unavailable. "
                "Retrieval quality may be degraded."
            )
            return

        model_sd = model.state_dict()
        repaired: dict[str, Any] = {}
        for key, value in state_dict.items():
            if "custom_text_proj" in key:
                continue  # already handled correctly by the package loader
            if key.startswith("model."):
                key = "base_model.model." + key[len("model.") :]
            if ".vision_tower.vision_model." in key:
                key = key.replace(".vision_tower.vision_model.", ".vision_tower.")

            target = model_sd.get(key)
            if target is not None and target.shape == value.shape:
                repaired[key] = value

        missing, unexpected = model.load_state_dict(repaired, strict=False)
        vision_loaded = sum(1 for k in repaired if "vision_tower" in k)
        vision_still_missing = [k for k in missing if "vision_tower" in k]

        if vision_still_missing:
            raise RuntimeError(
                f"Vision tower still has {len(vision_still_missing)} missing "
                f"weights after repair ({len(unexpected)} unmatched checkpoint "
                "keys). The installed sauerkrautlm-colpali package and the "
                f"'{settings.colpali_model_name}' checkpoint are incompatible."
            )
        logger.info(
            "Weight repair: applied %d tensors (%d vision tower). "
            "Still missing=%d, unexpected=%d.",
            len(repaired),
            vision_loaded,
            len(missing),
            len(unexpected),
        )

    def _warmup(self) -> None:
        """Run one tiny forward pass so CUDA kernels get initialised."""
        assert self._model is not None and self._processor is not None
        started = time.perf_counter()
        try:
            batch = self._processor.process_queries(["warmup"]).to(self._device)
            with torch.no_grad():
                self._model(**batch)
            logger.info("Model warmup done in %.2fs.", time.perf_counter() - started)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Model warmup failed (non-fatal): %s", exc)

    @staticmethod
    def _prepare_page(image: Image.Image) -> Image.Image:
        """Downscale a rendered page so it fits a single vision tile.

        Pages are rendered at ~96 DPI (e.g. 816x1056) but the vision encoder
        works on 512px tiles; feeding the full-size image triggers multi-tile
        splitting (measured 8x slower for no retrieval benefit on papers).
        """
        max_side = settings.image_max_side
        if not max_side or max(image.size) <= max_side:
            return image.convert("RGB")
        scale = max_side / float(max(image.size))
        new_size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
        return image.convert("RGB").resize(new_size, Image.LANCZOS)

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------
    def ingest_document(
        self,
        filename: str,
        pages: Iterator[tuple[int, int, str, Image.Image]],
        progress_cb: Callable[[int, int], None] | None = None,
    ) -> int:
        """Embed and store a document page-by-page from a streaming iterator.

        Each page is embedded and immediately persisted to ChromaDB before
        the next one is requested, so peak memory stays at roughly one
        rendered page + one embedding regardless of document length.

        Args:
            filename: Source PDF filename.
            pages: Iterator yielding (page_number, total_pages, text, image),
                as produced by ``pdf_parser.iter_pdf_pages``.
            progress_cb: Optional callback invoked as (pages_done, total_pages)
                after each page is stored.

        Returns:
            Number of pages ingested.
        """
        self._ensure_model()
        assert self._model is not None and self._processor is not None

        done = 0
        for page_number, total_pages, text, image in pages:
            # Fit the page into a single vision tile (big speedup)
            image = self._prepare_page(image)
            # Process a single image to minimize memory
            batch = self._processor.process_images([image]).to(self._device)
            with torch.no_grad():
                emb = self._model(**batch)
            # Multi-vector output: (1, num_tokens, dim) -> mean-pool to one vector
            pooled = emb.mean(dim=1).cpu().tolist()[0]
            embedding = [float(v) for v in pooled]

            self._collection.upsert(
                ids=[f"{filename}::page::{page_number}"],
                embeddings=[embedding],
                documents=[text],
                metadatas=[
                    {
                        "filename": filename,
                        "page": page_number,
                        "total_pages": total_pages,
                    }
                ],
            )

            done += 1
            if progress_cb is not None:
                progress_cb(done, total_pages)

            # Free the big tensors eagerly. Expensive global cleanups
            # (gc.collect / cuda.empty_cache) are throttled to every
            # `embed_gc_interval` pages instead of running after every page.
            del batch, emb, pooled, embedding, image
            if done % settings.embed_gc_interval == 0:
                self._free_memory()

        self._free_memory()
        logger.info("Ingested %d pages from %s", done, filename)
        return done

    def _embed_query(self, query: str) -> list[float]:
        """Compute an embedding for a text query."""
        self._ensure_model()
        assert self._model is not None and self._processor is not None

        batch = self._processor.process_queries([query]).to(self._device)
        with torch.no_grad():
            embedding = self._model(**batch)
        pooled = embedding.mean(dim=1).cpu().tolist()[0]
        return list(map(float, pooled))

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