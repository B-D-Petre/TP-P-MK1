import pandas as pd

from poc_trustpilot.Pipeline.config import SAMPLE_SIZE, SILVER
from poc_trustpilot.Pipeline.sample import star_quotas, stratified_sample


def test_star_quotas_proportional_and_exact():
    stars = pd.Series([5] * 70 + [1] * 16 + [4] * 7 + [3] * 4 + [2] * 3)
    q = star_quotas(stars, 1000)
    assert sum(q.values()) == 1000
    assert q == {1: 160, 2: 30, 3: 40, 4: 70, 5: 700}
    assert sum(star_quotas(stars, 7).values()) == 7


def test_sample_matches_dataset_star_mix():
    path = SILVER / "reviews_clean.parquet"
    if not path.exists():
        return  # silver layer not built yet
    reviews = pd.read_parquet(path)
    s = stratified_sample(reviews, SAMPLE_SIZE)
    assert len(s) == SAMPLE_SIZE and s["review_id"].is_unique
    assert not s["is_low_content"].any()
    expected = reviews["stars"].value_counts(normalize=True)
    got = s["stars"].value_counts(normalize=True)
    assert (got - expected).abs().max() < 0.01
