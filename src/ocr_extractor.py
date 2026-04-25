"""
ocr_extractor.py
─────────────────
Extracts hidden text from image-based spam attachments using OCR.

Background (from your paper, Section 1):
  Modern spammers embed text inside images to evade keyword-based filters.
  Standard TF-IDF classifiers are completely blind to this technique.

Pipeline:
  Image bytes → PIL pre-processing → Tesseract OCR → clean_text() → TF-IDF

BUG FIX (v2):
  - extract_and_clean() previously used "from src.preprocessor import clean_text"
    which caused a circular/path import error when called from within the src/
    package. Fixed to use a relative-safe absolute import that always resolves
    correctly regardless of how the module is imported.

Dependencies:
  pip install pytesseract Pillow
  System binary:
    Linux/WSL:  sudo apt-get install tesseract-ocr
    macOS:      brew install tesseract
    Windows:    https://github.com/UB-Mannheim/tesseract/wiki
  Streamlit Cloud: add tesseract-ocr to packages.txt (already included)
"""

import io
import logging
import sys
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)

# ── Optional-import guard ─────────────────────────────────────────────────────
try:
    from PIL import Image, ImageEnhance, ImageFilter
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("Pillow not installed — image OCR unavailable. pip install Pillow")

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logger.warning("pytesseract not installed — image OCR unavailable. pip install pytesseract")


ImageInput = Union[str, Path, bytes]


# ── Image loading ─────────────────────────────────────────────────────────────

def _load_image(image_input: ImageInput):
    """Load an image from a file path, raw bytes, or file-like object."""
    if not PIL_AVAILABLE:
        raise RuntimeError("Pillow is not installed.")

    if isinstance(image_input, (str, Path)):
        return Image.open(image_input)
    elif isinstance(image_input, bytes):
        return Image.open(io.BytesIO(image_input))
    elif hasattr(image_input, "read"):
        # Streamlit UploadedFile or any file-like object
        return Image.open(image_input)
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")


# ── Image pre-processing for OCR ──────────────────────────────────────────────

def _preprocess_for_ocr(img):
    """
    Enhance image quality before Tesseract OCR.

    Steps applied:
      1. Convert to RGB       → consistent channel handling (RGBA/P-mode safe)
      2. Convert to greyscale → OCR works on luminance only
      3. Upscale if tiny      → Tesseract needs >= ~300px width for accuracy
      4. Boost contrast       → spam images often use low-contrast text
      5. Sharpen              → counteracts JPEG compression blur
    """
    img = img.convert("RGB")
    img = img.convert("L")                         # greyscale

    min_width = 600
    if img.width < min_width:
        scale    = min_width / img.width
        new_size = (int(img.width * scale), int(img.height * scale))
        img = img.resize(new_size, Image.LANCZOS)
        logger.debug("Upscaled image to %s for OCR", new_size)

    img = ImageEnhance.Contrast(img).enhance(2.0)  # contrast boost
    img = img.filter(ImageFilter.SHARPEN)           # sharpening
    return img


# ── Core OCR ─────────────────────────────────────────────────────────────────

def extract_text_from_image(
    image_input: ImageInput,
    lang: str = "eng",
    config: str = "--psm 6",
) -> str:
    """
    Run Tesseract OCR on an image and return the raw extracted text.

    Args:
        image_input: File path, bytes, or file-like object.
        lang:        Tesseract language code ('eng' for English spam).
        config:      Tesseract page segmentation mode.
                     '--psm 6'  = uniform block of text (default).
                     '--psm 11' = sparse text (better for scattered lines).

    Returns:
        Raw OCR text string (may be empty if no text found).
    """
    if not PIL_AVAILABLE or not TESSERACT_AVAILABLE:
        raise RuntimeError(
            "OCR requires: pip install Pillow pytesseract  "
            "AND the Tesseract system binary."
        )

    try:
        img = _load_image(image_input)
        img = _preprocess_for_ocr(img)
        raw_text: str = pytesseract.image_to_string(img, lang=lang, config=config)
        logger.info("OCR extracted %d characters.", len(raw_text))
        return raw_text.strip()
    except Exception as exc:
        logger.error("OCR failed: %s", exc)
        return ""   # Fail gracefully — app.py handles empty string


def is_ocr_available() -> bool:
    """
    Runtime check: True if Pillow + pytesseract + Tesseract binary are all present.
    Used by app.py to conditionally show the image-upload tab.
    """
    if not PIL_AVAILABLE or not TESSERACT_AVAILABLE:
        return False
    try:
        pytesseract.get_tesseract_version()
        return True
    except pytesseract.TesseractNotFoundError:
        return False


# ── Public convenience function ───────────────────────────────────────────────

def extract_and_clean(image_input: ImageInput) -> str:
    """
    Full pipeline: Image → OCR → preprocess text → return cleaned string.

    The cleaned text is compatible with the same TF-IDF vectoriser used for
    email bodies, so the multimodal pipeline is seamless.

    BUG FIX: Import is done here at call time (not at module level) to avoid
    circular import issues when both modules are inside the src/ package.
    The import path 'src.preprocessor' resolves correctly because train.py
    and app.py both add the project root to sys.path before any src imports.
    """
    # Lazy import avoids circular dependency at module load time
    from src.preprocessor import clean_text   # noqa: PLC0415

    raw = extract_text_from_image(image_input)
    if not raw:
        return ""
    return clean_text(raw)
