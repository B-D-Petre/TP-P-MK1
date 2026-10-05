"""Layer 2: knowledge graph construction and Leiden community detection (macro, then micro inside each macro)."""
from itertools import combinations

import igraph as ig
import leidenalg
import networkx as nx
import pandas as pd

from .config import HUB_ENTITIES, LEIDEN_MACRO_RESOLUTION, LEIDEN_MICRO_RESOLUTION, SEED

UNCLUSTERED = -1


def relation_table(triplets: pd.DataFrame) -> pd.DataFrame:
    """Directed, predicate-typed relations aggregated over reviews."""
    if triplets.empty:
        return pd.DataFrame(columns=["source_id", "predicate", "target_id", "weight", "sentiment_mix", "review_ids"])
    g = triplets.groupby(["subject_id", "predicate", "object_id"])
    rel = g.agg(weight=("review_id", "size"), review_ids=("review_id", lambda s: sorted(set(s))),
                sentiment_mix=("sentiment", lambda s: s.value_counts().to_dict())).reset_index()
    return rel.rename(columns={"subject_id": "source_id", "object_id": "target_id"})


def build_graph(mentions: pd.DataFrame, triplets: pd.DataFrame, entities: pd.DataFrame) -> nx.Graph:
    """Undirected weighted entity graph: relation edges (weight 2 each) + co-mention edges within a review
    (1/(k-1) per pair, so a review mentioning many entities doesn't dominate)."""
    G = nx.Graph()
    for e in entities.itertuples(index=False):
        G.add_node(e.entity_id, name=e.name)

    def bump(a, b, w, kind):
        if a == b:
            return
        if G.has_edge(a, b):
            G[a][b]["weight"] += w
            G[a][b]["kinds"].add(kind)
        else:
            G.add_edge(a, b, weight=w, kinds={kind})

    for t in triplets.itertuples(index=False):
        bump(t.subject_id, t.object_id, 2.0, t.predicate)

    per_review = pd.concat([
        mentions[["review_id", "entity_id"]],
        triplets[["review_id", "subject_id"]].rename(columns={"subject_id": "entity_id"}),
        triplets[["review_id", "object_id"]].rename(columns={"object_id": "entity_id"}),
    ]).drop_duplicates()
    for _, ids in per_review.groupby("review_id")["entity_id"]:
        ids = sorted(set(ids))
        if len(ids) > 1:
            for a, b in combinations(ids, 2):
                bump(a, b, 1.0 / (len(ids) - 1), "co_mention")
    return G


def _leiden(G: nx.Graph, resolution: float) -> dict[str, int]:
    if G.number_of_edges() == 0:
        return {n: i for i, n in enumerate(G.nodes)}
    g = ig.Graph.from_networkx(G)
    part = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition, weights="weight",
                                    resolution_parameter=resolution, seed=SEED)
    return {g.vs[v]["_nx_name"]: c for c, members in enumerate(part) for v in members}


def _renumber_by_size(labels: dict[str, int]) -> dict[str, int]:
    sizes = pd.Series(labels).value_counts()
    order = {old: new for new, old in enumerate(sorted(sizes.index, key=lambda c: (-sizes[c], c)))}
    return {n: order[c] for n, c in labels.items()}


def detect_communities(G: nx.Graph, hub_ids: set[str]) -> pd.DataFrame:
    """Return node -> macro_cluster, micro_cluster. Hubs and isolated nodes get UNCLUSTERED (-1).
    Micro clusters are numbered '<macro>.<k>' so each nests inside exactly one macro cluster."""
    core = G.subgraph([n for n in G.nodes if n not in hub_ids]).copy()
    core.remove_nodes_from([n for n in list(core.nodes) if core.degree(n) == 0])

    macro = _renumber_by_size(_leiden(core, LEIDEN_MACRO_RESOLUTION)) if core.number_of_nodes() else {}
    micro: dict[str, str] = {}
    for c in sorted(set(macro.values())):
        members = [n for n, m in macro.items() if m == c]
        sub = core.subgraph(members)
        labels = _renumber_by_size(_leiden(sub, LEIDEN_MICRO_RESOLUTION)) if len(members) > 3 else {n: 0 for n in members}
        micro.update({n: f"{c}.{k}" for n, k in labels.items()})

    return pd.DataFrame({
        "entity_id": list(G.nodes),
        "macro_cluster": [macro.get(n, UNCLUSTERED) for n in G.nodes],
        "micro_cluster": [micro.get(n, str(UNCLUSTERED)) for n in G.nodes],
        "degree": [G.degree(n) for n in G.nodes],
    })


def hub_entity_ids(entities: pd.DataFrame) -> set[str]:
    return set(entities.loc[entities["name"].isin(HUB_ENTITIES), "entity_id"])


def pagerank(G: nx.Graph) -> dict[str, float]:
    if not G.number_of_edges():
        return {n: 0.0 for n in G.nodes}
    g = ig.Graph.from_networkx(G)
    return dict(zip(g.vs["_nx_name"], g.pagerank(weights="weight")))
