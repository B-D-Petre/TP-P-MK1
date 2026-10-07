"""Deterministic stratified sample of reviews to send to the LLM."""
import numpy as np
import pandas as pd

from .config import SEED

MIN_WORDS = 8


def star_quotas(stars: pd.Series, n: int) -> dict[int, int]:
    """n split across star levels in proportion to their share of the data (largest-remainder rounding)."""
    share = stars.value_counts(normalize=True).sort_index()
    raw = share * n
    quotas = np.floor(raw).astype(int)
    for s in (raw - quotas).sort_values(ascending=False).index[: n - quotas.sum()]:
        quotas[s] += 1
    return quotas.to_dict()


def stratified_sample(reviews: pd.DataFrame, n: int, seed: int = SEED) -> pd.DataFrame:
    """Pick n informative reviews whose star mix matches the full dataset, spread over language x source.

    Star quotas come from all reviews; rows are drawn only from informative ones (not low-content, >= MIN_WORDS).
    Within a star level every (language group, source group) stratum is shuffled and ranked; rows are taken in
    order of (rank + 0.5) / stratum size, which gives a proportional spread that is stable for a given seed.
    """
    if n >= len(reviews):  # full run: every review with at least one word, short ones included
        return reviews[reviews["n_words"] > 0].sort_values(["stars", "review_id"]).reset_index(drop=True)
    pool = reviews[~reviews["is_low_content"] & (reviews["n_words"] >= MIN_WORDS)].copy()
    top_langs = pool["language"].value_counts().head(4).index
    pool["lang_group"] = pool["language"].where(pool["language"].isin(top_langs), "other")

    rng = np.random.default_rng(seed)
    picked = []
    for star, k in star_quotas(reviews["stars"], n).items():
        b = pool[pool["stars"] == star]
        if k <= 0 or b.empty:
            continue
        b = b.assign(_r=rng.random(len(b))).sort_values("_r")
        strata = b.groupby(["lang_group", "source_group"])
        # Midpoint rank: a stratum of size s contributes its i-th row at (i + 0.5) / s, so tiny strata don't
        # jump the queue the way a plain i / s would (every stratum's first row would tie at 0).
        b["_order"] = (strata.cumcount() + 0.5) / strata["review_id"].transform("size") + b["_r"] * 1e-9
        picked.append(b.sort_values("_order").head(k))
    out = pd.concat(picked).drop(columns=["_r", "_order"])
    return out.sort_values(["stars", "review_id"]).reset_index(drop=True)
