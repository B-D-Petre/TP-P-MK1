"""Layer 2: turn each Leiden community into a named, human-readable user interest."""
import hashlib

import pandas as pd

from .clean import truncate_text
from .config import (MAX_OUTPUT_TOKENS_SUMMARY, MAX_QUOTE_CHARS, MAX_SUMMARY_ENTITIES, MAX_SUMMARY_QUOTES, MODEL,
                     PROMPT_VERSION, SILVER)
from .llm import JsonlCache, LLMError, parse
from .schema import InterestSummary

SYSTEM_PROMPT = """You name and describe a "user interest": a cluster of related concepts that a community-detection \
algorithm found in a knowledge graph built from customer reviews of Trustpilot (the review platform).

You receive the cluster's concepts (with category, number of mentions and average star rating of the reviews \
mentioning them) and verbatim evidence quotes, possibly in several languages.

- `title`: 2-6 words naming what users care about, e.g. "Removal of Genuine Reviews", "Easy Pre-Purchase Research".
- `summary`: 2-3 sentences (at most 60 words) in English, grounded only in the evidence: what users experience or \
want, and why it matters.
- `sentiment_label`: the dominant sentiment of users in this cluster.
- `key_entities`: up to 5 concept names copied exactly from the input.
Never include personal data or company names from the quotes."""


def _payload(entities: pd.DataFrame, quotes: list[str], n_reviews: int, avg_stars: float) -> str:
    """Capped input: top entities only, a few quotes, each quote shortened."""
    lines = [f"Cluster stats: {n_reviews} reviews, average rating {avg_stars:.1f}/5", "", "Concepts:"]
    for e in entities.head(MAX_SUMMARY_ENTITIES).itertuples(index=False):
        lines.append(f"- {e.name} [{e.category}] mentions={e.mention_count} avg_stars={e.avg_stars:.1f}")
    lines += ["", "Evidence quotes:"] + [f'- "{truncate_text(q, MAX_QUOTE_CHARS)}"' for q in quotes[:MAX_SUMMARY_QUOTES]]
    return "\n".join(lines)


def summarize_cluster(entities: pd.DataFrame, quotes: list[str], n_reviews: int, avg_stars: float,
                      cache: JsonlCache) -> dict | None:
    payload = _payload(entities, quotes, n_reviews, avg_stars)
    key = hashlib.sha256(f"{PROMPT_VERSION}|{MODEL}|{payload}".encode()).hexdigest()
    hit = cache.get(key)
    if hit:
        return hit["summary"]
    try:
        result = parse(SYSTEM_PROMPT, payload, InterestSummary,
                       max_tokens=MAX_OUTPUT_TOKENS_SUMMARY).model_dump(mode="json")
    except LLMError as e:
        print(f"  ! summary failed: {e}")
        return None
    cache.put(key, {"summary": result})
    return result


def open_cache() -> JsonlCache:
    return JsonlCache(SILVER / "summaries.jsonl")


def pick_quotes(mentions: pd.DataFrame, entity_ids: list[str], k: int = MAX_SUMMARY_QUOTES) -> list[str]:
    """Up to k evidence quotes from distinct reviews, most-mentioned entities first, mixing sentiments."""
    m = mentions[mentions["entity_id"].isin(entity_ids)].copy()
    rank = m["entity_id"].map(m["entity_id"].value_counts())
    m = m.assign(_rank=-rank).sort_values(["_rank", "sentiment", "review_id"]).drop_duplicates("review_id")
    return [q for q in m["evidence"].head(k) if isinstance(q, str) and q.strip()]

