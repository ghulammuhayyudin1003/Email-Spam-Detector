"""
preprocessor.py
───────────────
Handles all text cleaning and normalisation before TF-IDF vectorisation.

Design principle (from your paper's Section 3):
  The dataset contains raw email text with HTML markup, URLs, special
  characters, and highly frequent but low-information words (stop words).
  Removing this noise is the single most important step for generalisation.

Author note: every function is intentionally small and single-responsibility
so you can unit-test, swap, or extend each step independently.
"""

import re
import string
import logging

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── One-time NLTK resource download (safe to run repeatedly) ──────────────────
def _download_nltk_resources() -> None:
    """Download required NLTK data files silently if not already present."""
    resources = {
        "tokenizers/punkt":     "punkt",
        "tokenizers/punkt_tab": "punkt_tab",
        "corpora/stopwords":    "stopwords",
        "corpora/wordnet":      "wordnet",
        "corpora/omw-1.4":      "omw-1.4",
    }
    for path, name in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            logger.info("Downloading NLTK resource: %s", name)
            nltk.download(name, quiet=True)

_download_nltk_resources()

# ── Module-level singletons (expensive to initialise; reuse across calls) ─────
_LEMMATIZER = WordNetLemmatizer()
_STOP_WORDS = set(stopwords.words("english"))

# Keep a handful of negations — they can flip sentiment/meaning in spam text
_STOP_WORDS.discard("not")
_STOP_WORDS.discard("no")
_STOP_WORDS.discard("nor")

# ── Individual cleaning steps ─────────────────────────────────────────────────

def remove_html_tags(text: str) -> str:
    """Strip HTML/XML markup. Spam often embeds instructions inside tags."""
    return re.sub(r"<[^>]+>", " ", text)


def remove_urls(text: str) -> str:
    """Remove URLs (http/https/www and bare domains).

    Rationale: URLs carry very little TF-IDF signal and inflate the
    vocabulary unnecessarily.
    """
    url_pattern = r"https?://\S+|www\.\S+|\S+\.(com|org|net|io|co)\S*"
    return re.sub(url_pattern, " ", text, flags=re.IGNORECASE)


def remove_email_addresses(text: str) -> str:
    """Remove email addresses to avoid spurious token proliferation."""
    return re.sub(r"\S+@\S+", " ", text)


def remove_punctuation(text: str) -> str:
    """Replace all punctuation with a space so 'hello!world' → 'hello world'."""
    translator = str.maketrans(string.punctuation, " " * len(string.punctuation))
    return text.translate(translator)


def remove_numbers(text: str) -> str:
    """Remove standalone numbers."""
    return re.sub(r"\b\d+\b", " ", text)


def normalise_whitespace(text: str) -> str:
    """Collapse any run of whitespace to a single space."""
    return re.sub(r"\s+", " ", text).strip()


def tokenise_and_lemmatise(text: str) -> str:
    """
    Tokenise → lower-case → remove stop words → lemmatise → rejoin.

    Lemmatisation (vs. stemming) produces real English words, which keeps
    the TF-IDF vocabulary human-readable.
    """
    tokens = nltk.word_tokenize(text.lower())
    cleaned = [
        _LEMMATIZER.lemmatize(token)
        for token in tokens
        if token not in _STOP_WORDS and len(token) > 2
    ]
    return " ".join(cleaned)


# ── Master pipeline ───────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Apply the full preprocessing pipeline to a single email string.

    Pipeline order:
      1. HTML removal      → avoids tags becoming garbage tokens
      2. URL removal       → before punctuation removal
      3. Email removal     → same reason
      4. Punctuation       → after URL/email so we don't mangle them first
      5. Numbers           → standalone digits only
      6. Tokenise/lemmatise → lower-case, stop-word removal, lemmatisation
      7. Whitespace        → final tidy-up

    Args:
        text: Raw email body string.

    Returns:
        Cleaned, lemmatised string ready for TF-IDF.
    """
    if not isinstance(text, str):
        return ""   # Gracefully handle NaN / None values from pandas

    text = remove_html_tags(text)
    text = remove_urls(text)
    text = remove_email_addresses(text)
    text = remove_punctuation(text)
    text = remove_numbers(text)
    text = tokenise_and_lemmatise(text)
    text = normalise_whitespace(text)
    return text


def clean_series(series) -> "pd.Series":   # type: ignore[name-defined]
    """
    Vectorised wrapper: apply clean_text to a pandas Series.

    Usage:
        df["clean_text"] = clean_series(df["text"])
    """
    return series.apply(clean_text)
