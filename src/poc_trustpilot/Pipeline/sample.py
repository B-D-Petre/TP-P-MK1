"""Deterministic stratified sample of reviews to send to the LLM."""
import numpy as np
import pandas as pd

from .config import SEED

# Negative reviews are where the specific, actionable interests live (notebook 01), so they are over-sampled
# relative to their 19% share of the data.
SENTIMENT_SHARE = {"negative": 0.44, "neutral": 0.12, "positive": 0.44}
MIN_WORDS = 8


def _bucket(stars: int) -> str:
    return "negative" if stars <= 2 else "neutral" if stars == 3 else "positive"


def stratified_sample(reviews: pd.DataFrame, n: int, seed: int = SEED) -> pd.DataFrame:
    """Pick n informative reviews, balanced by sentiment bucket and spread over language x source within each.

    Within a bucket every (language group, source group) stratum is shuffled and ranked; rows are then taken in
    order of (rank + 0.5) / stratum size, which gives a proportional spread that is stable for a given seed.
    """
    pool = reviews[~reviews["is_low_content"] & (reviews["n_words"] >= MIN_WORDS)].copy()
    pool["bucket"] = pool["stars"].map(_bucket)
    top_langs = pool["language"].value_counts().head(4).index
    pool["lang_group"] = pool["language"].where(pool["language"].isin(top_langs), "other")

    quotas = {b: int(round(n * s)) for b, s in SENTIMENT_SHARE.items()}
    quotas["positive"] = n - quotas["negative"] - quotas["neutral"]

    rng = np.random.default_rng(seed)
    picked = []
    for bucket, k in quotas.items():
        b = pool[pool["bucket"] == bucket]
        if k <= 0 or b.empty:
            continue
        b = b.assign(_r=rng.random(len(b))).sort_values("_r")
        strata = b.groupby(["lang_group", "source_group"])
        # Midpoint rank: a stratum of size s contributes its i-th row at (i + 0.5) / s, so tiny strata don't
        # jump the queue the way a plain i / s would (every stratum's first row would tie at 0).
        b["_order"] = (strata.cumcount() + 0.5) / strata["review_id"].transform("size") + b["_r"] * 1e-9
        picked.append(b.sort_values("_order").head(k))
    out = pd.concat(picked).drop(columns=["_r", "_order"])
    return out.sort_values(["bucket", "review_id"]).reset_index(drop=True)
