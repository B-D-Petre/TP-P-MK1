"""Orchestrates bronze -> silver -> extraction -> graph -> interests -> gold."""
import time

from .bronze import build_bronze
from .canonicalize import canonicalize
from .clean import build_silver_reviews
from .config import GOLD, SAMPLE_SIZE, SILVER
from .extract import extract_reviews
from .gold import build_gold, entity_stats
from .graph import build_graph, detect_communities, hub_entity_ids, pagerank, relation_table
from .llm import METER
from .sample import stratified_sample


def run(sample_size: int = SAMPLE_SIZE) -> None:
    t0 = time.time()
    print("[1/6] bronze: load raw CSV")
    bronze = build_bronze()
    print(f"  {len(bronze):,} reviews")

    print("[2/6] silver: clean text + stratified sample")
    silver = build_silver_reviews(bronze)
    sample = stratified_sample(silver, sample_size)
    sample.to_parquet(SILVER / "sample.parquet", index=False)
    print(f"  sample of {len(sample)}: stars {sample['stars'].value_counts().sort_index().to_dict()}, "
          f"languages {sample['language'].value_counts().to_dict()}")

    print(f"[3/6] layer 1: LLM joint extraction")
    extractions = extract_reviews(sample)
    ok = extractions["extraction"].notna().sum()
    print(f"  {ok}/{len(extractions)} reviews extracted")
    if ok == 0:
        raise SystemExit("No extractions available - nothing to build.")

    print("[4/6] layer 1: canonicalization")
    mentions, triplets, entities, lineage = canonicalize(extractions)
    print(f"  {len(mentions)} aspect mentions, {len(triplets)} triplets | {lineage['raw'].nunique()} raw forms -> "
          f"{lineage['normalized'].nunique()} normalized -> {len(entities)} canonical entities")

    print("[5/6] layer 2: knowledge graph + Leiden")
    G = build_graph(mentions, triplets, entities)
    clusters = detect_communities(G, hub_entity_ids(entities))
    ents = entity_stats(entities, mentions, triplets, sample, clusters, pagerank(G))
    n_macro = clusters.loc[clusters["macro_cluster"] >= 0, "macro_cluster"].nunique()
    n_micro = clusters.loc[clusters["micro_cluster"] != "-1", "micro_cluster"].nunique()
    print(f"  graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges | {n_macro} macro / {n_micro} micro clusters")

    print("[6/6] layer 2 -> gold: interest summaries + golden tables")
    tables = build_gold(sample, extractions, mentions, triplets, ents, relation_table(triplets), G, lineage)
    for name, df in tables.items():
        print(f"  {name}: {len(df)} rows")
    print(f"\nLLM usage this run: {METER.report()}")
    print(f"Done in {time.time() - t0:.0f}s -> {GOLD}")
