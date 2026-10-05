"""Golden layer: enriched, analysis-ready tables + a graph export for the UI."""
import json

import pandas as pd

from .config import GOLD, MIN_INTEREST_ENTITIES
from .graph import UNCLUSTERED
from .summarize import open_cache, pick_quotes, summarize_cluster


def entity_stats(entities: pd.DataFrame, mentions: pd.DataFrame, triplets: pd.DataFrame,
                 reviews: pd.DataFrame, clusters: pd.DataFrame, pr: dict) -> pd.DataFrame:
    stars = reviews.set_index("review_id")["stars"]
    touches = pd.concat([
        mentions[["review_id", "entity_id", "sentiment"]],
        triplets[["review_id", "subject_id", "sentiment"]].rename(columns={"subject_id": "entity_id"}),
        triplets[["review_id", "object_id", "sentiment"]].rename(columns={"object_id": "entity_id"}),
    ])
    touches["stars"] = touches["review_id"].map(stars)
    agg = touches.groupby("entity_id").agg(
        mention_count=("review_id", "size"),
        review_count=("review_id", "nunique"),
        avg_stars=("stars", "mean"),
        pct_negative=("sentiment", lambda s: (s == "negative").mean()),
    )
    out = entities.merge(agg, left_on="entity_id", right_index=True, how="left").merge(clusters, on="entity_id")
    out["pagerank"] = out["entity_id"].map(pr).fillna(0.0)
    return out.fillna({"mention_count": 0, "review_count": 0}).astype({"mention_count": int, "review_count": int})


def entity_reviews(mentions: pd.DataFrame, triplets: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([
        mentions[["review_id", "entity_id"]],
        triplets[["review_id", "subject_id"]].rename(columns={"subject_id": "entity_id"}),
        triplets[["review_id", "object_id"]].rename(columns={"object_id": "entity_id"}),
    ]).drop_duplicates()


def build_interests(ents: pd.DataFrame, mentions: pd.DataFrame, triplets: pd.DataFrame,
                    reviews: pd.DataFrame) -> pd.DataFrame:
    """One row per macro and micro community with >= MIN_INTEREST_ENTITIES entities, summarised by the LLM."""
    cache = open_cache()
    er = entity_reviews(mentions, triplets)
    rv = reviews.set_index("review_id")
    rows = []
    for level, col in [("macro", "macro_cluster"), ("micro", "micro_cluster")]:
        for cid, members in ents[ents[col].astype(str) != str(UNCLUSTERED)].groupby(col):
            if len(members) < MIN_INTEREST_ENTITIES:
                continue
            parent = str(cid).split(".")[0] if level == "micro" else None
            if level == "micro" and (ents["macro_cluster"].astype(str) == parent).sum() == len(members):
                continue  # the micro cluster is the whole macro cluster - nothing new to summarise
            ids = list(members["entity_id"])
            rids = sorted(set(er.loc[er["entity_id"].isin(ids), "review_id"]))
            r = rv.loc[rids]
            top = members.sort_values(["mention_count", "pagerank"], ascending=False)
            quotes = pick_quotes(mentions, ids)
            summary = summarize_cluster(top.head(15), quotes, len(rids), r["stars"].mean(), cache) or {}
            rows.append({
                "interest_id": f"{level}_{cid}",
                "level": level,
                "cluster": str(cid),
                "parent_interest_id": f"macro_{parent}" if parent is not None else None,
                "title": summary.get("title", f"Cluster {cid}"),
                "summary": summary.get("summary", ""),
                "sentiment_label": summary.get("sentiment_label", "mixed"),
                "key_entities": summary.get("key_entities", []),
                "n_entities": len(members),
                "n_reviews": len(rids),
                "avg_stars": round(float(r["stars"].mean()), 2),
                "pct_negative_reviews": round(float((r["stars"] <= 2).mean()), 3),
                "top_entities": list(top["name"].head(10)),
                "representative_quotes": quotes[:5],
                "language_mix": r["language"].value_counts().to_dict(),
                "source_mix": r["source_group"].value_counts().to_dict(),
            })
    return pd.DataFrame(rows)


def review_interest_bridge(ents: pd.DataFrame, mentions: pd.DataFrame, triplets: pd.DataFrame,
                           interests: pd.DataFrame) -> pd.DataFrame:
    """review x interest, weight = share of the review's clustered entities that fall in the interest."""
    er = entity_reviews(mentions, triplets).merge(ents[["entity_id", "macro_cluster", "micro_cluster"]], on="entity_id")
    known = set(interests["interest_id"])
    parts = []
    for level, col in [("macro", "macro_cluster"), ("micro", "micro_cluster")]:
        x = er[er[col].astype(str) != str(UNCLUSTERED)].assign(interest_id=lambda d: level + "_" + d[col].astype(str))
        x = x[x["interest_id"].isin(known)]
        w = x.groupby(["review_id", "interest_id"]).size().rename("n").reset_index()
        w["weight"] = (w["n"] / w.groupby("review_id")["n"].transform("sum")).round(3)
        parts.append(w.assign(level=level)[["review_id", "interest_id", "level", "weight"]])
    return pd.concat(parts, ignore_index=True)


def build_gold(reviews: pd.DataFrame, extractions: pd.DataFrame, mentions: pd.DataFrame, triplets: pd.DataFrame,
               ents: pd.DataFrame, relations: pd.DataFrame, G) -> dict[str, pd.DataFrame]:
    interests = build_interests(ents, mentions, triplets, reviews)
    bridge = review_interest_bridge(ents, mentions, triplets, interests)

    ext = extractions.set_index("review_id")["extraction"]
    gr = reviews.copy()
    gr["overall_sentiment"] = gr["review_id"].map(lambda r: (ext.get(r) or {}).get("overall_sentiment"))
    gr["is_about_trustpilot"] = gr["review_id"].map(lambda r: (ext.get(r) or {}).get("is_about_trustpilot"))
    gr["n_aspects"] = gr["review_id"].map(mentions.groupby("review_id").size()).fillna(0).astype(int)
    primary = (bridge[bridge["level"] == "macro"].sort_values("weight", ascending=False)
               .drop_duplicates("review_id").set_index("review_id")["interest_id"])
    gr["primary_macro_interest_id"] = gr["review_id"].map(primary)
    gr["extraction_ok"] = gr["review_id"].map(lambda r: ext.get(r) is not None)
    gr = gr.drop(columns=["bucket", "lang_group"], errors="ignore")

    tables = {
        "gold_reviews": gr,
        "gold_aspect_mentions": mentions[["review_id", "entity_id", "category", "opinion", "sentiment", "evidence"]],
        "gold_entities": ents,
        "gold_relations": relations.assign(sentiment_mix=relations["sentiment_mix"].map(json.dumps)),
        "gold_interests": interests.assign(language_mix=interests["language_mix"].map(json.dumps),
                                           source_mix=interests["source_mix"].map(json.dumps)),
        "gold_review_interests": bridge,
    }
    GOLD.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_parquet(GOLD / f"{name}.parquet", index=False)

    graph = {
        "nodes": [{k: (v.item() if hasattr(v, "item") else v) for k, v in rec.items()}
                  for rec in ents.drop(columns=["aliases"]).assign(aliases=ents["aliases"].map(list)).to_dict("records")],
        "edges": [{"source": a, "target": b, "weight": round(d["weight"], 3), "kinds": sorted(d["kinds"])}
                  for a, b, d in G.edges(data=True)],
    }
    (GOLD / "graph.json").write_text(json.dumps(graph, ensure_ascii=False, default=str), encoding="utf-8")
    return tables
