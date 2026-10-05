"""Silver layer, part 1: text normalisation driven by the findings in notebooks/02.

Only transformations that are safe for every language are applied here. Language-sensitive steps
(accents, stemming, stopwords) are deliberately left to the LLM, which reads the review in its own language.
"""
import html
import re
import unicodedata

import pandas as pd
from rapidfuzz import fuzz

from .config import SILVER, SOURCE_GROUP

TOKEN_RE = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)*")

_CHAR_MAP = str.maketrans({
    "’": "'", "‘": "'", "´": "'", "`": "'",
    "“": '"', "”": '"', "„": '"', "«": '"', "»": '"',
    "…": "...", " ": " ", "–": "-", "—": "-",
})
_ZERO_WIDTH_RE = re.compile("[​-‏﻿]")

# Brand variants (section 5 of notebook 02): split / hyphenated / inflected / domain forms...
_BRAND_SPLIT_RE = re.compile(r"(?i)\btrustpilot\.(?:com|net|dk|de|fr|co\.uk)\b|\btrust[\s-]*pil+o?t(?:e|en|s|'s)?\b")
# ...the uppercase abbreviation...
_BRAND_ABBREV_RE = re.compile(r"\bTP\b")
# ...and single-token misspellings (Truspilot, Trustpliot, turstpilot), matched fuzzily below.
_BRAND_CANDIDATE_RE = re.compile(r"\b[Tt][^\W\d_]{6,12}\b")


def normalize_text(text: str) -> str:
    """Decode HTML entities, unify typography and whitespace. Keeps accents, case and line breaks."""
    text = unicodedata.normalize("NFC", html.unescape(text or ""))
    text = _ZERO_WIDTH_RE.sub("", text.translate(_CHAR_MAP))
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return re.sub(r"\n{2,}", "\n", "\n".join(lines)).strip()


def canonicalize_brand(text: str) -> str:
    """Map every spelling of the brand to 'Trustpilot'."""
    text = _BRAND_SPLIT_RE.sub("Trustpilot", text)
    text = _BRAND_ABBREV_RE.sub("Trustpilot", text)

    def fix(m: re.Match) -> str:
        word = m.group(0)
        return "Trustpilot" if word.lower() != "trustpilot" and fuzz.ratio(word.lower(), "trustpilot") >= 85 else word

    return _BRAND_CANDIDATE_RE.sub(fix, text)


def merge_title(title: str, body: str) -> tuple[str, bool]:
    """Return (text for extraction, whether the title was kept).

    Titles are usually a copy or an auto-truncated prefix of the body (~77% in notebook 02); using both would
    count the same words twice. A title is kept only when it adds words the body doesn't have.
    """
    t, b = title.strip(), body.strip()
    if not t:
        return b, False
    if not b:
        return t, True
    stem = t.rstrip(".").rstrip().casefold()
    if t.casefold() == b.casefold() or (stem and b.casefold().startswith(stem)):
        return b, False
    if set(TOKEN_RE.findall(t.casefold())) <= set(TOKEN_RE.findall(b.casefold())):
        return b, False
    return f"{t}\n{b}", True


def truncate_text(text: str, max_chars: int) -> str:
    """Cut text to at most max_chars at a word boundary, marking the cut with ' [...]'."""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars].rsplit(None, 1)[0] if " " in text[:max_chars] else text[:max_chars]
    return cut.rstrip(" ,;:-") + " [...]"


def is_low_content(text: str) -> bool:
    """Fewer than 4 word tokens: 'Top!', 'Excellent service', emoji-only, dots-only."""
    return len(TOKEN_RE.findall(text)) < 4


def build_silver_reviews(bronze: pd.DataFrame) -> pd.DataFrame:
    df = bronze[["review_id", "created_at", "stars", "source", "language"]].copy()
    df["source_group"] = df["source"].map(SOURCE_GROUP).fillna("other")
    df["title_clean"] = bronze["title"].map(normalize_text).map(canonicalize_brand)
    df["body_clean"] = bronze["text"].map(normalize_text).map(canonicalize_brand)
    merged = [merge_title(t, b) for t, b in zip(df["title_clean"], df["body_clean"])]
    df["text_for_llm"] = [m[0] for m in merged]
    df["title_kept"] = [m[1] for m in merged]
    df["n_words"] = df["text_for_llm"].map(lambda s: len(TOKEN_RE.findall(s)))
    df["is_low_content"] = df["text_for_llm"].map(is_low_content)

    SILVER.mkdir(parents=True, exist_ok=True)
    df.to_parquet(SILVER / "reviews_clean.parquet", index=False)
    return df
