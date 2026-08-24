"""PDF parsing utilities: streaming text extraction and page rendering.

Everything goes through pypdfium2 in a SINGLE pass: per page we extract text
and render an image, hand both to the caller, then release the page's native
resources immediately. Peak memory is roughly ONE rendered page instead of
the whole document, which matters on 8 GB machines.

(The previous implementation opened the PDF twice — pypdf for text and
pypdfium2 for images — and materialised every page image up front.)
"""

import logging
from pathlib import Path
from typing import BinaryIO, Iterator

import pypdfium2 as pdfium  # type: ignore
from PIL import Image

logger = logging.getLogger(__name__)


class PDFParseError(Exception):
    """Raised when a PDF cannot be parsed."""


def iter_pdf_pages(
    source: str | Path | BinaryIO, dpi: int = 96
) -> Iterator[tuple[int, int, str, Image.Image]]:
    """Yield ``(page_number, total_pages, text, image)`` one page at a time.

    Args:
        source: Path to a PDF file or a binary file-like object.
        dpi: Rendering resolution. 96 DPI (~1.33x) keeps memory low on
            resource-constrained machines while preserving layout for ColLFM2.

    Yields:
        Tuples of (1-based page number, total page count, extracted text,
        rendered PIL image).

    Raises:
        PDFParseError: If the PDF cannot be opened or processed.
    """
    try:
        pdf = pdfium.PdfDocument(source)
    except Exception as exc:  # noqa: BLE001
        raise PDFParseError(f"Failed to open PDF: {exc}") from exc

    total = len(pdf)
    if total == 0:
        pdf.close()
        return

    scale = dpi / 72.0
    try:
        for index in range(total):
            page = pdf[index]
            try:
                # --- Text (same pass, no second library needed) ---
                text = ""
                try:
                    textpage = page.get_textpage()
                    try:
                        text = textpage.get_text_range() or ""
                    finally:
                        textpage.close()
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Text extraction failed on page %d: %s", index + 1, exc
                    )

                # --- Image ---
                bitmap = page.render(scale=scale)
                image = bitmap.to_pil()  # copies pixels out of the native buffer
                del bitmap  # release pdfium's pixel buffer right away
            finally:
                page.close()

            yield index + 1, total, text, image
            # `image` is dropped by the caller's loop; nothing accumulates.
    except PDFParseError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise PDFParseError(f"Failed while processing pages: {exc}") from exc
    finally:
        pdf.close()


def count_pages(source: str | Path | BinaryIO) -> int:
    """Return the number of pages in a PDF without processing it."""
    try:
        pdf = pdfium.PdfDocument(source)
        try:
            return len(pdf)
        finally:
            pdf.close()
    except Exception as exc:  # noqa: BLE001
        raise PDFParseError(f"Failed to open PDF: {exc}") from exc