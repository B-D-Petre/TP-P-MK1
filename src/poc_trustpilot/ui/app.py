"""Track 2 - visual exploration of the enriched knowledge base.

Run with:  uv run streamlit run src/poc_trustpilot/ui/app.py
"""
import json
import math
import os
from itertools import combinations
from pathlib import Path

import altair as alt
import networkx as nx
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network

from poc_trustpilot.Pipeline.config import GOLD as DEFAULT_GOLD

GOLD = Path(os.environ.get("POC_GOLD_DIR", DEFAULT_GOLD))

# Fixed categorical order (validated reference palette); clusters beyond 8 fold into "Other".
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
OTHER, HUB, UNCLUSTERED = "#b4b2a9", "#52514e", "#dedcd5"
SENTIMENT_COLORS = {"negative (1-2★)": "#e34948", "neutral (3★)": "#c3c2b7", "positive (4-5★)": "#2a78d6"}
SURFACE, INK = "#fcfcfb", "#0b0b0b"

st.set_page_config(page_title="Trustpilot Interest Graph", page_icon="🕸️", layout="wide")


@st.cache_data
def load():
    t = {name: pd.read_parquet(GOLD / f"{name}.parquet") for name in
         ["gold_reviews", "gold_aspect_mentions", "gold_entities", "gold_relations", "gold_interests",
          "gold_review_interests"]}
    t["graph"] = json.loads((GOLD / "graph.json").read_text(encoding="utf-8"))
    return t


if not (GOLD / "graph.json").exists():
    st.error(f"No golden layer found in `{GOLD}`. Run the pipeline first:  `uv run poc-trustpilot run`")
    st.stop()

data = load()
reviews, mentions, entities = data["gold_reviews"], data["gold_aspect_mentions"], data["gold_entities"]
relations, interests, bridge = data["gold_relations"], data["gold_interests"], data["gold_review_interests"]
hub_names = {"trustpilot", "reviewer", "business"}

macro_int = interests[interests["level"] == "macro"].sort_values("n_reviews", ascending=False)
micro_int = interests[interests["level"] == "micro"]


def cluster_colors(level: str) -> dict[str, str]:
    """Cluster id -> color. Ids are numbered by size in the pipeline, so colors follow the cluster, not the filter."""
    col = "macro_cluster" if level == "macro" else "micro_cluster"
    sizes = entities[entities[col].astype(str) != "-1"].groupby(entities[col].astype(str)).size()
    ranked = sorted(sizes.index, key=lambda c: (-sizes[c], c))
    return {c: (PALETTE[i] if i < len(PALETTE) else OTHER) for i, c in enumerate(ranked)}


def cluster_titles(level: str) -> dict[str, str]:
    """Cluster id -> interest title. Clusters too small to be summarised get a generic label."""
    macro_titles = dict(zip(macro_int["cluster"], macro_int["title"]))
    if level == "macro":
        return {c: macro_titles.get(c, f"Minor cluster {c}")
                for c in entities["macro_cluster"].astype(str).unique()}
    titles = dict(zip(micro_int["cluster"], micro_int["title"]))
    for c in entities["micro_cluster"].astype(str).unique():
        titles.setdefault(c, macro_titles.get(c.split(".")[0], f"Minor cluster {c}"))
    return titles


@st.cache_data
def graph_layout(nodes: tuple, edges: tuple, clusters: tuple) -> dict[str, tuple[float, float]]:
    """Two-level layout so each Leiden cluster reads as a region: a spring layout inside every cluster, and the
    clusters themselves placed by a spring layout of the cluster-level graph (related clusters end up close)."""
    H = nx.Graph()
    H.add_nodes_from(nodes)
    H.add_weighted_edges_from(edges)
    group_of = dict(clusters)
    groups: dict[str, list] = {}
    for n in nodes:
        groups.setdefault(group_of.get(n, "-1"), []).append(n)

    meta = nx.Graph()
    meta.add_nodes_from(groups)
    for a, b, w in edges:
        ga, gb = group_of.get(a, "-1"), group_of.get(b, "-1")
        if ga != gb:
            meta.add_edge(ga, gb, weight=meta.get_edge_data(ga, gb, {"weight": 0})["weight"] + w)
    centers = nx.spring_layout(meta, seed=7, weight="weight", k=2.2 / math.sqrt(max(len(groups), 1)), iterations=300)
    spread = 260 * math.sqrt(len(groups))

    pos = {}
    for g, ms in groups.items():
        radius = 70 * math.sqrt(len(ms))
        local = (nx.spring_layout(H.subgraph(ms), seed=7, weight="weight", iterations=200)
                 if len(ms) > 1 else {ms[0]: (0.0, 0.0)})
        cx, cy = centers[g]
        for n, (x, y) in local.items():
            pos[n] = (float(cx) * spread + float(x) * radius, float(cy) * spread + float(y) * radius)
    return pos


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Filters")
    stars_sel = st.multiselect("Star rating", [1, 2, 3, 4, 5], default=[1, 2, 3, 4, 5])
    src_sel = st.multiselect("Source", sorted(reviews["source_group"].unique()),
                             default=sorted(reviews["source_group"].unique()))
    lang_sel = st.multiselect("Language", sorted(reviews["language"].unique()),
                              default=sorted(reviews["language"].unique()))
    st.header("Graph")
    level = st.radio("Color nodes by Leiden level", ["macro", "micro"], horizontal=True,
                     format_func=lambda s: f"{s} interests")
    focus = st.selectbox("Highlight interest", ["(all)"] + list(macro_int["title"]))
    min_mentions = st.slider("Min. mentions per entity", 1, max(2, int(entities["mention_count"].max())), 1)
    show_hubs = st.checkbox("Show hub nodes (Trustpilot, reviewer, business)", value=False)
    show_unclustered = st.checkbox("Show unclustered entities", value=False)

rv = reviews[reviews["stars"].isin(stars_sel) & reviews["source_group"].isin(src_sel) & reviews["language"].isin(lang_sel)]
rids = set(rv["review_id"])

# entity -> reviews touching it (aspect mentions + relation endpoints)
touch = pd.concat([
    mentions[["review_id", "entity_id"]],
    relations[["review_ids", "source_id"]].explode("review_ids").rename(columns={"review_ids": "review_id", "source_id": "entity_id"}),
    relations[["review_ids", "target_id"]].explode("review_ids").rename(columns={"review_ids": "review_id", "target_id": "entity_id"}),
]).drop_duplicates()
visible_ents = set(touch.loc[touch["review_id"].isin(rids), "entity_id"])

# ---------------------------------------------------------------- header
st.title("Trustpilot reviews: interest knowledge graph")
st.caption("Layer 1 LLM extraction → canonical entities → Leiden communities → LLM-named interests. "
           f"Golden layer: `{GOLD}`")
k = st.columns(5)
k[0].metric("Reviews", f"{len(rv)} / {len(reviews)}")
k[1].metric("Entities", len(entities))
k[2].metric("Relations", len(relations))
k[3].metric("Macro interests", len(macro_int))
k[4].metric("Avg rating", f"{rv['stars'].mean():.2f}★" if len(rv) else "–")

tab_int, tab_graph, tab_rev = st.tabs(["Interests", "Knowledge graph", "Reviews"])

# ---------------------------------------------------------------- interests
with tab_int:
    macro_colors = cluster_colors("macro")
    b = bridge[(bridge["level"] == "macro") & bridge["review_id"].isin(rids)].merge(
        reviews[["review_id", "stars"]], on="review_id")
    b["sentiment"] = pd.cut(b["stars"], [0, 2, 3, 5], labels=list(SENTIMENT_COLORS)).astype(str)
    b = b.merge(macro_int[["interest_id", "title"]], on="interest_id")
    if b.empty:
        st.info("No reviews match the filters.")
    else:
        counts = b.groupby(["title", "sentiment"]).size().rename("reviews").reset_index()
        order = list(b.groupby("title").size().sort_values(ascending=False).index)
        chart = alt.Chart(counts).mark_bar(cornerRadiusEnd=3).encode(
            y=alt.Y("title:N", sort=order, title=None, axis=alt.Axis(labelLimit=320)),
            x=alt.X("reviews:Q", title="reviews (a review can touch several interests)"),
            color=alt.Color("sentiment:N", scale=alt.Scale(domain=list(SENTIMENT_COLORS), range=list(SENTIMENT_COLORS.values())),
                            legend=alt.Legend(orient="top", title=None)),
            order=alt.Order("sentiment:N"),
            tooltip=["title", "sentiment", "reviews"],
        # Step-based height: a fixed pixel height (spec or st.altair_chart) squeezes all bands into one row in Streamlit.
        ).properties(height=alt.Step(34)).configure_scale(bandPaddingInner=0.45)
        st.markdown("##### Main interests by number of reviews")
        st.altair_chart(chart, width="stretch")

    for it in macro_int.itertuples(index=False):
        if focus != "(all)" and it.title != focus:
            continue
        n_here = b.loc[b["interest_id"] == it.interest_id, "review_id"].nunique() if not b.empty else 0
        with st.container(border=True):
            color = macro_colors.get(it.cluster, OTHER)
            st.markdown(f"<span style='display:inline-block;width:12px;height:12px;border-radius:3px;"
                        f"background:{color};margin-right:8px'></span>**{it.title}** · _{it.sentiment_label}_",
                        unsafe_allow_html=True)
            c = st.columns(4)
            c[0].metric("Reviews (filtered)", f"{n_here} / {it.n_reviews}")
            c[1].metric("Avg rating", f"{it.avg_stars:.2f}★")
            c[2].metric("Negative reviews", f"{it.pct_negative_reviews:.0%}")
            c[3].metric("Entities", it.n_entities)
            st.write(it.summary)
            st.caption("Top entities: " + ", ".join(it.top_entities[:8]))
            if len(it.representative_quotes):
                st.markdown(f"> {it.representative_quotes[0]}")
            subs = micro_int[micro_int["parent_interest_id"] == it.interest_id]
            if len(subs):
                with st.expander(f"{len(subs)} micro interests"):
                    for s in subs.sort_values("n_reviews", ascending=False).itertuples(index=False):
                        st.markdown(f"**{s.title}** ({s.n_reviews} reviews, {s.avg_stars:.1f}★) — {s.summary}")

# ---------------------------------------------------------------- graph
with tab_graph:
    colors, titles = cluster_colors(level), cluster_titles(level)
    col = "macro_cluster" if level == "macro" else "micro_cluster"
    focus_cluster = None
    if focus != "(all)":
        focus_cluster = macro_int.loc[macro_int["title"] == focus, "cluster"].iloc[0]

    ents = entities[entities["entity_id"].isin(visible_ents) & (entities["mention_count"] >= min_mentions)].copy()
    ents["is_hub"] = ents["name"].isin(hub_names)
    ents["cluster"] = ents[col].astype(str)
    if not show_hubs:
        ents = ents[~ents["is_hub"]]
    if not show_unclustered:
        ents = ents[(ents["cluster"] != "-1") | ents["is_hub"]]
    keep = set(ents["entity_id"])
    vis_edges = [e for e in data["graph"]["edges"] if e["source"] in keep and e["target"] in keep]
    pos = graph_layout(tuple(sorted(keep)), tuple((e["source"], e["target"], e["weight"]) for e in vis_edges),
                       tuple(zip(ents["entity_id"], ents["cluster"])))

    net = Network(height="680px", width="100%", bgcolor=SURFACE, font_color=INK, cdn_resources="remote")
    for e in ents.itertuples(index=False):
        if e.is_hub:
            color = HUB
        elif e.cluster == "-1":
            color = UNCLUSTERED
        else:
            color = colors.get(e.cluster, OTHER)
        faded = focus_cluster is not None and str(e.macro_cluster) != str(focus_cluster) and not e.is_hub
        tip = (f"{e.name}\ncategory: {e.category}\ninterest: {titles.get(e.cluster, 'unclustered')}\n"
               f"mentions: {e.mention_count} in {e.review_count} reviews\navg rating: {e.avg_stars:.1f}★ | "
               f"negative: {e.pct_negative:.0%}" + (f"\naliases: {', '.join(e.aliases[:6])}" if len(e.aliases) else ""))
        x, y = pos[e.entity_id]
        net.add_node(e.entity_id, label=e.name, title=tip, size=8 + 6 * math.sqrt(e.mention_count), x=x, y=y,
                     color={"background": color, "border": SURFACE, "highlight": {"background": color, "border": INK}},
                     opacity=0.25 if faded else 1.0, borderWidth=2,
                     font={"size": 18, "color": "#89878199" if faded else INK})
    rel_pairs = {tuple(sorted(p)) for p in zip(relations["source_id"], relations["target_id"])}
    for edge in vis_edges:
        a, b2 = edge["source"], edge["target"]
        is_rel = tuple(sorted((a, b2))) in rel_pairs
        net.add_edge(a, b2, value=edge["weight"], title=", ".join(edge["kinds"]),
                     color="#898781" if is_rel else "#d6d4cc")
    # Positions come from Python (deterministic, no drift); physics off, so the initial view fits the whole graph.
    net.set_options(json.dumps({
        "physics": {"enabled": False},
        "edges": {"smooth": False, "scaling": {"min": 1, "max": 6}},
        "interaction": {"hover": True, "tooltipDelay": 120, "dragNodes": True},
    }))
    graph_html = net.generate_html().replace(
        "</body>", "<script>setTimeout(function () { network.fit(); }, 200);"
                   " window.addEventListener('resize', function () { network.fit(); });</script></body>")

    left, right = st.columns([4, 1])
    with left:
        if ents.empty:
            st.info("No entities match the filters.")
        else:
            components.html(graph_html, height=700)
    with right:
        st.markdown(f"**Leiden {level} clusters**")
        shown = ents[ents["cluster"] != "-1"].groupby("cluster").size().sort_values(ascending=False)
        legend_items = [(c, n) for c, n in shown.items() if colors.get(c) and colors[c] != OTHER]
        for c, n in legend_items:
            st.markdown(f"<span style='display:inline-block;width:11px;height:11px;border-radius:3px;"
                        f"background:{colors[c]};margin-right:6px'></span>{titles.get(c, c)} <small>({n})</small>",
                        unsafe_allow_html=True)
        if any(colors.get(c) == OTHER for c in shown.index):
            st.markdown(f"<span style='display:inline-block;width:11px;height:11px;border-radius:3px;background:{OTHER};"
                        f"margin-right:6px'></span>Other clusters", unsafe_allow_html=True)
        if show_hubs:
            st.markdown(f"<span style='display:inline-block;width:11px;height:11px;border-radius:3px;background:{HUB};"
                        f"margin-right:6px'></span>Hub (not clustered)", unsafe_allow_html=True)
        if show_unclustered:
            st.markdown(f"<span style='display:inline-block;width:11px;height:11px;border-radius:3px;"
                        f"background:{UNCLUSTERED};margin-right:6px'></span>Unclustered", unsafe_allow_html=True)
        st.caption("Node size = mentions. Dark edges = extracted relations, light edges = co-mentioned in a review. "
                   "Hover a node for details; drag to rearrange.")

# ---------------------------------------------------------------- reviews
with tab_rev:
    titles_macro = dict(zip(macro_int["interest_id"], macro_int["title"]))
    view = rv.assign(interest=rv["primary_macro_interest_id"].map(titles_macro))
    if focus != "(all)":
        ids = set(bridge.loc[bridge["interest_id"] == macro_int.loc[macro_int["title"] == focus, "interest_id"].iloc[0],
                             "review_id"])
        view = view[view["review_id"].isin(ids)]
    st.dataframe(view[["review_id", "stars", "language", "source_group", "overall_sentiment", "interest",
                       "n_aspects", "text_for_llm"]].sort_values("stars"),
                 width="stretch", hide_index=True,
                 column_config={"text_for_llm": st.column_config.TextColumn("review text", width="large")})
    pick = st.selectbox("Inspect extracted aspects for a review", ["–"] + list(view["review_id"]))
    if pick != "–":
        st.write(view.loc[view["review_id"] == pick, "text_for_llm"].iloc[0])
        st.dataframe(mentions[mentions["review_id"] == pick].merge(entities[["entity_id", "name"]], on="entity_id")
                     [["name", "category", "opinion", "sentiment", "evidence"]], width="stretch", hide_index=True)
