"""Optional step - OCR for pages that are pictures (scanned PDFs).

Uses EasyOCR (Persian + English). It is imported only when a page really needs
OCR, so typed PDFs never load it. OCR models are stored in models/easyocr.
"""
import logging
import re
import threading

import pymupdf

from config import (MODELS_DIR, OCR_DPI, OCR_LANGUAGES, OCR_MAX_PAGES,
                    OCR_MIN_IMAGE_COVERAGE, OCR_USE_GPU)

log = logging.getLogger(__name__)


class OCRError(Exception):
    """Raised when OCR is not installed, cannot start, or the PDF is too long to OCR."""


_reader = None
_reader_lock = threading.Lock()

_RTL = re.compile(r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]")
_LATIN = re.compile(r"[A-Za-z]")


def page_looks_scanned(page) -> bool:
    """True when pictures cover a big part of the page (a scan, not a logo)."""
    try:
        page_area = page.rect.width * page.rect.height
        if page_area <= 0:
            return False
        covered = 0.0
        for info in page.get_image_info():
            part = pymupdf.Rect(info["bbox"]) & page.rect
            if not part.is_empty:
                covered += part.width * part.height
        return covered / page_area >= OCR_MIN_IMAGE_COVERAGE
    except Exception:
        return False


def _get_reader():
    """Load EasyOCR once (the first call downloads the models)."""
    global _reader
    with _reader_lock:
        if _reader is not None:
            return _reader
        try:
            import easyocr
        except ImportError as exc:
            raise OCRError(
                "OCR support is not installed. Run Start.bat again to install it "
                "(or: pip install -r backend/requirements-ocr.txt)."
            ) from exc
        except Exception as exc:
            raise OCRError(f"OCR support is installed but could not start: {exc}") from exc

        model_dir = MODELS_DIR / "easyocr"
        model_dir.mkdir(parents=True, exist_ok=True)
        try:
            _reader = easyocr.Reader(
                OCR_LANGUAGES,
                gpu=OCR_USE_GPU,
                model_storage_directory=str(model_dir),
            )
        except Exception as exc:
            raise OCRError(
                f"OCR models could not be loaded or downloaded: {exc}. "
                "The first scanned PDF needs internet access (GitHub) to download them."
            ) from exc
        return _reader


def _is_rtl(text: str) -> bool:
    rtl = len(_RTL.findall(text))
    return rtl > 0 and rtl >= len(_LATIN.findall(text))


def _boxes_to_text(results) -> str:
    """Turn EasyOCR boxes into lines of text in reading order (RTL-aware)."""
    items = []
    for box, text, _conf in results:
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        items.append({"x0": min(xs), "y0": min(ys), "y1": max(ys), "text": text})
    items.sort(key=lambda i: (i["y0"] + i["y1"]) / 2)

    lines = []
    for it in items:
        cy = (it["y0"] + it["y1"]) / 2
        h = it["y1"] - it["y0"]
        if lines and abs(cy - lines[-1]["cy"]) <= 0.5 * max(h, lines[-1]["h"]):
            line = lines[-1]
            line["items"].append(it)
            line["cy"] = sum((i["y0"] + i["y1"]) / 2 for i in line["items"]) / len(line["items"])
            line["h"] = max(line["h"], h)
        else:
            lines.append({"cy": cy, "h": h, "items": [it]})

    out = []
    for line in lines:
        rtl = _is_rtl(" ".join(i["text"] for i in line["items"]))
        line["items"].sort(key=lambda i: i["x0"], reverse=rtl)  # Persian reads right to left
        out.append(" ".join(i["text"] for i in line["items"]))
    return "\n".join(out)


def ocr_pages(doc, page_numbers: list[int]) -> list[dict]:
    """OCR the given 1-based pages. Returns [{'page': n, 'text': '...'}, ...]."""
    if len(page_numbers) > OCR_MAX_PAGES:
        raise OCRError(
            f"{len(page_numbers)} pages need OCR but the limit is {OCR_MAX_PAGES}. "
            "Raise OCR_MAX_PAGES in backend/config.py."
        )
    reader = _get_reader()

    results = []
    for number in page_numbers:
        try:
            pixmap = doc[number - 1].get_pixmap(dpi=OCR_DPI)
            raw = reader.readtext(pixmap.tobytes("png"), detail=1, paragraph=False)
        except Exception as exc:
            raise OCRError(f"OCR failed on page {number}: {exc}") from exc
        text = _boxes_to_text(raw)
        if text.strip():
            results.append({"page": number, "text": text})
    return results
