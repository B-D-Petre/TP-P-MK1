import pandas as pd

from poc_trustpilot.Pipeline.insights import classify, share_trend, with_month
from poc_trustpilot.Pipeline.report import badge_class


def test_classify_thresholds():
    assert classify(0.10, 0.10, 100) == "stable"
    assert classify(0.10, 0.13, 100) == "rising"          # +3 pts and +30%
    assert classify(0.10, 0.07, 100) == "falling"
    assert classify(0.30, 0.34, 100) == "stable"          # +4 pts but only +13% relative
    assert classify(0.010, 0.014, 100) == "stable"        # +40% relative but only +0.4 pts
    assert classify(0.10, 0.30, 5) == "too few reviews"


def test_share_trend_uses_monthly_share_and_skips_thin_months():
    months = [f"2021-{m:02d}-15" for m in range(7, 13) for _ in range(40)] + ["2022-01-02"]  # last month has 1 review
    reviews = with_month(pd.DataFrame({"review_id": [f"r{i}" for i in range(len(months))],
                                       "created_at": pd.to_datetime(months, utc=True)}))
    # topic in 4 of 40 reviews in Jul-Sep, 8 of 40 in Oct-Dec -> 10% -> 20% share
    hits = [f"r{m * 40 + i}" for m in range(6) for i in range(4 if m < 3 else 8)]
    share, s = share_trend(pd.Series(hits), reviews)
    assert list(share.index) == ["2021-07", "2021-08", "2021-09", "2021-10", "2021-11", "2021-12"]
    assert round(s["early_share"], 3) == 0.10 and round(s["late_share"], 3) == 0.20
    assert s["trend"] == "rising" and round(s["change_pp"], 1) == 10.0


def test_badge_colour_follows_meaning():
    assert badge_class("rising", "negative") == "bad"   # more complaints
    assert badge_class("rising", "positive") == "good"  # more praise
    assert badge_class("falling", "negative") == "good"
    assert badge_class("rising", None) == "flat"
    assert badge_class("stable", "negative") == "flat"
