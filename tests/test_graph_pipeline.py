import pandas as pd
import pytest

from poc_trustpilot.Pipeline import gold
from poc_trustpilot.Pipeline.canonicalize import build_canonical_map, canonicalize, normalize_name, normalize_trace
from poc_trustpilot.Pipeline.graph import build_graph, detect_communities, hub_entity_ids, pagerank, relation_table


def test_normalize_name():
    assert normalize_name("Fake Reviews") == "fake review"
    assert normalize_name("Trust Pilot") == "trustpilot"
    assert normalize_name("the business") == "business"
    assert normalize_name("Companies") == "company"
    assert normalize_name("e-mail") == "e mail"


def test_canonical_map_merges_variants_but_not_distinct_concepts():
    m = build_canonical_map(["email", "email", "e mail", "positive review", "negative review"])
    assert m["e mail"] == "email"
    assert m["positive review"] != m["negative review"]


def test_normalize_trace_ends_in_normalize_name():
    trace = normalize_trace("The Trust Pilot Reviews")
    assert trace[0] == ("raw LLM output", "The Trust Pilot Reviews")
    assert trace[-1][1] == normalize_name("The Trust Pilot Reviews") == "trustpilot review"


def test_lineage_records_exact_and_fuzzy_merges():
    ext = pd.DataFrame({"review_id": ["r1", "r2", "r3"], "extraction": [
        {"aspects": [_aspect("Email", "account_verification"), _aspect("emails", "account_verification")],
         "triplets": [], "overall_sentiment": "negative", "is_about_trustpilot": True},
        {"aspects": [_aspect("email", "account_verification")],
         "triplets": [_triplet("trustpilot", "HAS_ISSUE", "e-mail")], "overall_sentiment": "negative",
         "is_about_trustpilot": True},
        None]})
    _, _, entities, lineage = canonicalize(ext)
    lin = lineage.set_index("raw")
    assert set(lin.loc[["Email", "emails", "email"], "canonical"]) == {"email"}
    assert lin.loc["emails", "merge"] == "exact"  # normalisation alone (plural) got it there
    assert lin.loc["e-mail", "merge"] == "fuzzy" and 90 <= lin.loc["e-mail", "fuzzy_score"] < 100
    assert lin.loc["e-mail", "roles"] == ["triplet object"]
    assert lineage["entity_id"].isin(entities["entity_id"]).all()


def _aspect(name, cat, sent="negative"):
    return {"aspect": name, "category": cat, "opinion": "x", "sentiment": sent, "evidence": f"quote about {name}"}


def _triplet(s, p, o):
    return {"subject": s, "predicate": p, "object": o, "sentiment": "negative"}


@pytest.fixture
def toy():
    # Two obvious themes (moderation problems / pre-purchase research) plus the brand hub.
    exts = []
    for i in range(6):
        exts.append({"aspects": [_aspect("review removal", "review_moderation"), _aspect("flagging", "review_moderation"),
                                 _aspect("Customer Support", "customer_support")],
                     "triplets": [_triplet("trustpilot", "HAS_ISSUE", "review removal"),
                                  _triplet("review removal", "CAUSED_BY", "flagging")],
                     "overall_sentiment": "negative", "is_about_trustpilot": True})
    for i in range(6):
        exts.append({"aspects": [_aspect("checking a shop", "research_before_buying", "positive"),
                                 _aspect("ease of use", "ease_of_use", "positive"),
                                 _aspect("honest reviews", "trust_credibility", "positive")],
                     "triplets": [_triplet("trustpilot", "USED_FOR", "checking a shop")],
                     "overall_sentiment": "positive", "is_about_trustpilot": True})
    ext = pd.DataFrame({"review_id": [f"r{i}" for i in range(12)], "extraction": exts})
    reviews = pd.DataFrame({"review_id": ext["review_id"], "stars": [1] * 6 + [5] * 6,
                            "language": "en", "source_group": ["organic"] * 6 + ["invited"] * 6})
    return ext, reviews


def test_graph_clusters_separate_themes_and_exclude_hub(toy):
    ext, _ = toy
    mentions, triplets, entities, lineage = canonicalize(ext)
    assert "customer support" in set(entities["name"])  # case/plural normalised
    G = build_graph(mentions, triplets, entities)
    hubs = hub_entity_ids(entities)
    cl = detect_communities(G, hubs).set_index("entity_id")["macro_cluster"]
    assert cl["e_trustpilot"] == -1
    assert cl["e_review_removal"] == cl["e_flagging"] != cl["e_ease_of_use"] == cl["e_honest_review"]
    # deterministic
    assert detect_communities(G, hubs).equals(detect_communities(G, hubs))


def test_gold_tables_without_api(toy, monkeypatch, tmp_path):
    ext, reviews = toy
    monkeypatch.setattr(gold, "GOLD", tmp_path)
    monkeypatch.setattr(gold, "open_cache", lambda: None)
    monkeypatch.setattr(gold, "summarize_cluster",
                        lambda ents, quotes, n, s, cache: {"title": "T", "summary": "S", "sentiment_label": "negative",
                                                           "key_entities": list(ents["name"][:2])})
    mentions, triplets, entities, lineage = canonicalize(ext)
    G = build_graph(mentions, triplets, entities)
    clusters = detect_communities(G, hub_entity_ids(entities))
    ents = gold.entity_stats(entities, mentions, triplets, reviews, clusters, pagerank(G))
    tables = gold.build_gold(reviews, ext, mentions, triplets, ents, relation_table(triplets), G, lineage)

    assert len(tables["gold_reviews"]) == 12
    assert set(tables["gold_interests"]["level"]) == {"macro"}  # micro == macro for these tiny clusters
    assert len(tables["gold_interests"]) == 2
    assert tables["gold_reviews"]["primary_macro_interest_id"].notna().all()
    assert (tmp_path / "graph.json").exists()
