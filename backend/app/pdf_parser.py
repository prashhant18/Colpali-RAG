"""PDF parsing utilities: text extraction and page-to-image conversion."""

import io
import logging
from pathlib import Path
from typing import BinaryIO

import pypdf
import pypdfium2 as pdfium  # type: ignore
from PIL import Image

logger = logging.getLogger(__name__)


class PDFParseError(Exception):
    """Raised when a PDF cannot be parsed."""


def extract_text(file_obj: BinaryIO) -> list[str]:
    """Extract text per page from a PDF file object.

    Args:
        file_obj: A binary file-like object containing the PDF.

    Returns:
        A list of strings, one per page (in order).

    Raises:
        PDFParseError: If the PDF cannot be read.
    """
    try:
        reader = pypdf.PdfReader(file_obj)
        pages: list[str] = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        return pages
    except Exception as exc:  # noqa: BLE001
        raise PDFParseError(f"Failed to extract text: {exc}") from exc


def render_page_images(file_obj: BinaryIO, dpi: int = 144) -> list[Image.Image]:
    """Render each page of a PDF to a PIL image.

    Uses pypdfium2 for fast, dependency-light rendering (no poppler required).

    Args:
        file_obj: A binary file-like object containing the PDF.
        dpi: Rendering resolution. 144 DPI (~2x) balances quality and size.

    Returns:
        A list of PIL Images, one per page.

    Raises:
        PDFParseError: If the PDF cannot be rendered.
    """
    try:
        file_obj.seek(0)
        pdf = pdfium.PdfDocument(file_obj)
        images: list[Image.Image] = []
        scale = dpi / 72.0
        for page in pdf:
            bitmap = page.render(scale=scale)
            pil_image = bitmap.to_pil()
            images.append(pil_image)
        pdf.close()
        return images
    except Exception as exc:  # noqa: BLE001
        raise PDFParseError(f"Failed to render pages: {exc}") from exc


def parse_pdf(file_obj: BinaryIO) -> tuple[list[str], list[Image.Image]]:
    """Parse a PDF into per-page text and images.

    Args:
        file_obj: A binary file-like object containing the PDF.

    Returns:
        A tuple of (page_texts, page_images).
    """
    file_obj.seek(0)
    texts = extract_text(file_obj)
    images = render_page_images(file_obj)
    if len(texts) != len(images):
        logger.warning(
            "Text/image page count mismatch: %d vs %d", len(texts), len(images)
        )
    return texts, images