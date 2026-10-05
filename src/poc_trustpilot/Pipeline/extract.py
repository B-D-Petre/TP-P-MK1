"""Layer 1: single-pass joint extraction of aspects and triplets from each review."""
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from .clean import truncate_text
from .config import (EVIDENCE_MAX_WORDS, MAX_ASPECTS, MAX_OUTPUT_TOKENS_EXTRACTION, MAX_REVIEW_CHARS, MAX_TRIPLETS,
                     MAX_WORKERS, MODEL, PROMPT_VERSION, SILVER)
from .llm import JsonlCache, LLMError, parse
from .schema import ReviewExtraction

SYSTEM_PROMPT = """You extract structured knowledge from customer reviews of Trustpilot (the review platform itself). \
The output feeds a knowledge graph in which one real-world concept must map to exactly one node, so naming \
consistency matters more than anything else.

## What to extract
1. **aspects** - the distinct things the reviewer has an opinion about: 1 to {max_aspects}, most important first.
   - `aspect`: a short **English**, lowercase, singular noun phrase, whatever the review's language
     ("review removal", "fake review", "email verification", "customer support", "invitation email").
     Prefer the generic concept over the specific instance ("negative review removal", not "my review of Acme").
   - `category`: the closest category from the enum. Use `other` only when nothing fits.
   - `opinion`: a short English phrase of at most 6 words ("removed without explanation", "easy to use").
   - `sentiment`: the reviewer's sentiment toward that aspect.
   - `evidence`: a verbatim quote from the review text, in its original language, at most {evidence_words} words.
2. **triplets** - explicit relationships the reviewer states between two concepts: 0 to {max_triplets}, using only \
the predicate enum. Subjects and objects follow the same naming rules as aspects, and should reuse the exact \
aspect names when they refer to the same thing. Use "trustpilot" for the platform and "business" for reviewed companies.
   - HAS_ISSUE: a concept has a problem ("trustpilot" HAS_ISSUE "review removal")
   - CAUSED_BY / LEADS_TO: cause and effect ("review removal" CAUSED_BY "automated flagging")
   - PRAISES / CRITICIZES: the reviewer's stance toward a concept, with "reviewer" as subject
   - REQUESTS: a feature or change the reviewer asks for ("reviewer" REQUESTS "photo upload")
   - USED_FOR: purpose ("trustpilot" USED_FOR "checking a shop before buying")
   - COMPARED_TO: explicit comparison with an alternative
3. **overall_sentiment** of the whole review.
4. **is_about_trustpilot**: false if the review is really about another company (a shop, a delivery) and was \
posted on Trustpilot's own page by mistake.

## Rules
- Never output personal data (names, emails, order numbers). Name third-party companies only as "business".
- Do not invent aspects that the text does not support. Short praise ("great site") still yields one aspect.
- Consistent naming across reviews: singular forms, no articles, no brand variants ("trustpilot" only).
- Be terse: the output is a compact record, not prose. Respect the limits above.

## Example
Review (de, 1 star): "Meine echte Bewertung wurde nach zwei Tagen gelöscht, angeblich wegen Verstoß. Keine Antwort vom Support."
- aspects: ("review removal", review_moderation, "genuine review deleted after two days", negative, \
"Meine echte Bewertung wurde nach zwei Tagen gelöscht"), ("customer support", customer_support, \
"no response", negative, "Keine Antwort vom Support")
- triplets: ("trustpilot", HAS_ISSUE, "review removal", negative), ("review removal", CAUSED_BY, \
"guideline violation claim", negative)
- overall_sentiment: negative; is_about_trustpilot: true
""".format(max_aspects=MAX_ASPECTS, max_triplets=MAX_TRIPLETS, evidence_words=EVIDENCE_MAX_WORDS)


def user_message(row) -> str:
    """Contextual string assembly: metadata header + review text (capped), no splitting."""
    return (f"Review metadata: stars={row.stars}/5 | source={row.source} ({row.source_group}) | "
            f"language={row.language} | date={row.created_at:%Y-%m-%d}\n\n"
            f"<review>\n{truncate_text(row.text_for_llm, MAX_REVIEW_CHARS)}\n</review>")


def _key(review_id: str) -> str:
    return f"{review_id}|{PROMPT_VERSION}|{MODEL}"


def extract_reviews(sample: pd.DataFrame) -> pd.DataFrame:
    """Return one row per sampled review with its extraction (dict) or an error. Uses and fills the cache."""
    cache = JsonlCache(SILVER / "extractions.jsonl")
    todo = [row for row in sample.itertuples(index=False) if cache.get(_key(row.review_id)) is None]
    print(f"  extraction: {len(sample) - len(todo)} cached, {len(todo)} to call")

    def work(row):
        try:
            result = parse(SYSTEM_PROMPT, user_message(row), ReviewExtraction, max_tokens=MAX_OUTPUT_TOKENS_EXTRACTION)
            result.aspects = result.aspects[:MAX_ASPECTS]  # enforce the caps even if the model overshoots
            result.triplets = result.triplets[:MAX_TRIPLETS]
            cache.put(_key(row.review_id), {"review_id": row.review_id, "extraction": result.model_dump(mode="json")})
        except LLMError as e:
            if "Authentication" in str(e):
                raise
            print(f"  ! {row.review_id}: {e}")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for fut in as_completed([pool.submit(work, r) for r in todo]):
            fut.result()

    rows = []
    for rid in sample["review_id"]:
        rec = cache.get(_key(rid))
        rows.append({"review_id": rid, "extraction": rec["extraction"] if rec else None})
    return pd.DataFrame(rows)
