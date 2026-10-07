"""Trends and stakeholder outputs computed from the golden layer only - no LLM calls, no pipeline re-run.

Volumes are expressed as a *share of that month's reviews*, so trends survive changes in sample size and in
monthly review volume. A trend compares the average share in the first 3 months with the last 3 months.
"""
from dataclasses import dataclass

import pandas as pd

from .config import GOLD, HUB_ENTITIES

MIN_MONTH_REVIEWS = 30   # months with fewer reviews (e.g. 1 review on 1 Feb 2022) are left out of trends
MIN_TREND_REVIEWS = 20   # below this, a trend is reported as "too few reviews"
WINDOW = 3               # months compared at each end
MIN_CHANGE_PP, MIN_CHANGE_REL = 0.5, 0.25  # rising/falling needs both: +-0.5 points of share and +-25% relative


def load_gold(gold=GOLD) -> dict[str, pd.DataFrame]:
    t = {n: pd.read_parquet(gold / f"gold_{n}.parquet")
         for n in ["reviews", "aspect_mentions", "entities", "relations", "interests", "review_interests"]}
    t["reviews"] = with_month(t["reviews"])
    return t


def with_month(reviews: pd.DataFrame) -> pd.DataFrame:
    return reviews.assign(month=reviews["created_at"].dt.tz_localize(None).dt.to_period("M").astype(str))


def valid_months(reviews: pd.DataFrame) -> pd.Series:
    """Reviews per month, only months with enough reviews to compute a share."""
    totals = reviews.groupby("month").size()
    return totals[totals >= MIN_MONTH_REVIEWS]


def classify(early: float, late: float, n: int) -> str:
    if n < MIN_TREND_REVIEWS:
        return "too few reviews"
    change = late - early
    if change >= MIN_CHANGE_PP / 100 and late >= early * (1 + MIN_CHANGE_REL):
        return "rising"
    if change <= -MIN_CHANGE_PP / 100 and late <= early * (1 - MIN_CHANGE_REL):
        return "falling"
    return "stable"


def share_trend(review_ids: pd.Series, reviews: pd.DataFrame) -> tuple[pd.Series, dict]:
    """Monthly share of reviews among `review_ids`, plus early/late/change/trend."""
    totals = valid_months(reviews)
    r = reviews[reviews["review_id"].isin(set(review_ids)) & reviews["month"].isin(totals.index)]
    share = (r.groupby("month")["review_id"].nunique().reindex(totals.index, fill_value=0) / totals)
    early, late = share.iloc[:WINDOW].mean(), share.iloc[-WINDOW:].mean()
    n = r["review_id"].nunique()
    return share, {"early_share": early, "late_share": late, "change_pp": (late - early) * 100,
                   "trend": classify(early, late, n), "n_reviews": n}


def interest_trends(t: dict, level: str = "macro") -> tuple[pd.DataFrame, pd.DataFrame]:
    """(monthly share per interest [interest x month], one-row-per-interest trend summary)."""
    reviews, bridge = t["reviews"], t["review_interests"]
    interests = t["interests"][t["interests"]["level"] == level]
    stars = reviews.set_index("review_id")["stars"]
    series, rows = {}, []
    for it in interests.itertuples(index=False):
        rids = bridge.loc[bridge["interest_id"] == it.interest_id, "review_id"]
        share, s = share_trend(rids, reviews)
        series[it.interest_id] = share
        rows.append({"interest_id": it.interest_id, "title": it.title, "sentiment_label": it.sentiment_label,
                     "pct_negative": float((stars.reindex(rids) <= 2).mean()) if len(rids) else 0.0, **s})
    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values("n_reviews", ascending=False).reset_index(drop=True)
    return pd.DataFrame(series).T, summary


# ---------------------------------------------------------------- stakeholders

@dataclass(frozen=True)
class Stakeholder:
    name: str
    question: str
    categories: tuple[str, ...] | None   # aspect categories that matter to them (None = everything)
    sentiment: str | None                 # only mentions with this sentiment (None = all)
    extra: str | None = None              # "requests" | "causes" | "channels"


STAKEHOLDERS = [
    Stakeholder("Leadership", "Where is sentiment heading, and what drives the negative reviews?",
                None, "negative", "channels"),
    Stakeholder("Trust & Safety", "Are complaints about moderation, fake reviews and scams growing?",
                ("review_moderation", "fake_reviews", "scam_fraud"), "negative"),
    Stakeholder("Product / UX", "Which product frictions do users hit, and what do they ask for?",
                ("ease_of_use", "account_verification", "emails_notifications"), "negative", "requests"),
    Stakeholder("Customer Support", "How is support experienced, and what causes the complaints?",
                ("customer_support",), None, "causes"),
    Stakeholder("B2B / Business customers", "What do businesses using Trustpilot think of tools and pricing?",
                ("business_tools_pricing",), None),
    Stakeholder("Marketing / Brand", "What do consumers value about Trustpilot, in their own words?",
                ("research_before_buying", "trust_credibility", "ease_of_use"), "positive"),
]


def stakeholder_view(t: dict, s: Stakeholder) -> dict:
    """Headline numbers, monthly share + trend, top entities, quotes and stakeholder-specific extras."""
    reviews, mentions, entities, relations = t["reviews"], t["aspect_mentions"], t["entities"], t["relations"]
    m = mentions
    if s.categories:
        m = m[m["category"].isin(s.categories)]
    if s.sentiment:
        m = m[m["sentiment"] == s.sentiment]
    m = m.merge(reviews[["review_id", "stars", "source_group", "language"]], on="review_id")
    name_of = dict(zip(entities["entity_id"], entities["name"]))
    hubs = {e for e, n in name_of.items() if n in HUB_ENTITIES}
    share, trend = share_trend(m["review_id"], reviews)
    rv = reviews[reviews["review_id"].isin(set(m["review_id"]))]

    top = (m[~m["entity_id"].isin(hubs)]
           .groupby("entity_id").agg(mentions=("review_id", "size"), reviews=("review_id", "nunique"),
                                     avg_stars=("stars", "mean"))
           .sort_values("mentions", ascending=False).head(8).reset_index())
    top.insert(0, "entity", top["entity_id"].map(name_of))
    # One quote per top entity, English first (stakeholder audience), most negative first within that.
    quotes = (m[m["entity_id"].isin(top["entity_id"].head(4))]
              .assign(_en=lambda d: d["language"] != "en").sort_values(["_en", "stars"])
              .drop_duplicates("review_id").drop_duplicates("entity_id")["evidence"].head(3).tolist())

    out = {"stakeholder": s, "n_reviews": rv["review_id"].nunique(),
           "share_of_reviews": rv["review_id"].nunique() / max(len(reviews), 1),
           "avg_stars": float(rv["stars"].mean()) if len(rv) else float("nan"),
           "organic_share": float((rv["source_group"] == "organic").mean()) if len(rv) else 0.0,
           "monthly_share": share, **trend, "top_entities": top.drop(columns="entity_id"), "quotes": quotes}

    if s.extra == "requests":
        req = relations[relations["predicate"] == "REQUESTS"].sort_values("weight", ascending=False)
        out["requests"] = [(name_of.get(x, x), int(w)) for x, w in zip(req["target_id"], req["weight"])][:8]
    elif s.extra == "causes":
        ids = set(m["entity_id"]) - hubs
        c = relations[relations["predicate"].isin(["CAUSED_BY", "LEADS_TO"])
                      & (relations["source_id"].isin(ids) | relations["target_id"].isin(ids))
                      & ~relations["source_id"].map(name_of).isin({"trustpilot", "reviewer"})]
        c = c.sort_values("weight", ascending=False).head(8)
        out["causes"] = [f"{name_of.get(a, a)} {p.replace('_', ' ').lower()} {name_of.get(b, b)}"
                         for a, p, b in zip(c["source_id"], c["predicate"], c["target_id"])]
    elif s.extra == "channels":
        totals = valid_months(reviews)
        r = reviews[reviews["month"].isin(totals.index)]
        out["stars_by_channel"] = r.groupby(["month", "source_group"])["stars"].mean().unstack()
        out["channel_stats"] = reviews.groupby("source_group").agg(
            reviews=("review_id", "size"), avg_stars=("stars", "mean"),
            pct_one_star=("stars", lambda x: (x == 1).mean()))
        _, it = interest_trends(t)
        if not it.empty:
            it = it.assign(negative_reviews=(it["n_reviews"] * it["pct_negative"]).round().astype(int))
            out["negative_drivers"] = it.sort_values("negative_reviews", ascending=False).head(5)
    return out


def headline(t: dict) -> dict:
    """Dataset-level numbers for the executive summary header."""
    r = t["reviews"]
    months = valid_months(r).index
    return {"n_reviews": len(r), "avg_stars": float(r["stars"].mean()),
            "pct_negative": float((r["stars"] <= 2).mean()), "organic_share": float((r["source_group"] == "organic").mean()),
            "n_macro_interests": int((t["interests"]["level"] == "macro").sum()),
            "n_entities": len(t["entities"]), "period": f"{months.min()} to {months.max()}" if len(months) else ""}
