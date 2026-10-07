"""Graphviz flowcharts of each pipeline stage for the UI's Pipeline tab. Numbers come from config, so they stay true."""
from poc_trustpilot.Pipeline import config as c
from poc_trustpilot.Pipeline.canonicalize import FUZZY_THRESHOLD
from poc_trustpilot.Pipeline.sample import MIN_WORDS

_HEAD = ('digraph {{ rankdir={rd}; bgcolor="transparent"; nodesep=0.3; ranksep=0.35; '
         'node [shape=box, style="rounded,filled", fillcolor="#f3f2ee", color="#c3c2b7", fontname="Helvetica", '
         'fontsize=11, fontcolor="#0b0b0b", margin="0.15,0.06"]; '
         'edge [color="#898781", fontname="Helvetica", fontsize=9, fontcolor="#52514e", arrowsize=0.7]; ')
LLM = 'fillcolor="#dbe9fb", color="#2a78d6"'          # Claude call
DATA = 'shape=cylinder, fillcolor="#ffffff"'          # file on disk
DECIDE = 'shape=diamond, style=filled, fillcolor="#fdf1d8", color="#eda100"'


def dot(body: str, rankdir: str = "LR") -> str:
    return _HEAD.format(rd=rankdir) + body + "}"


LEGEND = dot(f'a [label="processing step"]; b [label="Claude API call", {LLM}]; c [label="file on disk", {DATA}]; '
             f'd [label="decision", {DECIDE}]; a -> b -> c -> d [style=invis];')

OVERVIEW = dot(f"""
csv [label="raw CSV\\n10,000 reviews", {DATA}];
bronze [label="1. Bronze\\nload + type"]; silver [label="2. Silver\\nclean + sample"];
ext [label="3. Layer 1\\nLLM extraction", {LLM}]; canon [label="4. Canonicalization\\nnormalize + fuzzy merge"];
kg [label="5. Layer 2\\ngraph + Leiden"]; summ [label="6. Interests\\nLLM summaries", {LLM}];
gold [label="golden layer\\ndata/gold/*", {DATA}]; ui [label="Streamlit UI"];
csv -> bronze -> silver -> ext -> canon -> kg -> summ -> gold -> ui;
""")

STAGES = {
    "1-2. Bronze & Silver: cleaning": (dot(f"""
csv [label="raw CSV", {DATA}]; load [label="read, parse dates,\\nmerge nb -> no"];
bronze [label="bronze/reviews.parquet", {DATA}];
unesc [label="html.unescape\\n(&amp;#229; -> å)"]; typo [label="typographic quotes, ´ -> ASCII,\\nwhitespace, zero-width"];
brand [label="brand variants -> Trustpilot\\n(regex + fuzzy >= 85)"];
title [label="title adds words\\nthe body lacks?", {DECIDE}];
both [label="text = title + body"]; body [label="text = body"];
low [label="flag low-content\\n(< 4 words)"]; silver [label="silver/reviews_clean.parquet", {DATA}];
csv -> load -> bronze -> unesc -> typo -> brand -> title; title -> both [label="yes (~22%)"];
title -> body [label="no"]; both -> low; body -> low; low -> silver;
""", "TB"), "Text normalisation from notebook 02. Accents, stemming and stopwords are deliberately **not** touched: "
          "the LLM reads each review in its own language."),

    "2. Sampling": (dot(f"""
silver [label="silver reviews", {DATA}]; elig [label="keep informative reviews\\n(not low-content, >= {MIN_WORDS} words)"];
quota [label="star quotas = dataset\\nstar distribution x {c.SAMPLE_SIZE}"];
strata [label="per star: strata =\\nlanguage x source"]; rank [label="shuffle (seed {c.SEED}),\\nmidpoint rank (i+0.5)/size"];
take [label="take top-k per star"]; sample [label="silver/sample.parquet", {DATA}];
silver -> elig -> quota -> strata -> rank -> take -> sample;
"""), "Deterministic: the same seed and data always give the same sample."),

    "3. Layer 1: LLM extraction": (dot(f"""
row [label="sampled review"]; hdr [label="metadata header +\\ntext capped at {c.MAX_REVIEW_CHARS} chars"];
cache [label="in cache?\\n(review_id, {c.PROMPT_VERSION}, model)", {DECIDE}];
hit [label="reuse cached\\nextraction"];
llm [label="{c.MODEL}\\nstructured output (Pydantic enums)\\nmax_tokens {c.MAX_OUTPUT_TOKENS_EXTRACTION}", {LLM}];
ok [label="refusal or\\nmax_tokens?", {DECIDE}]; skip [label="log + skip\\n(retried next run)"];
trim [label="trim to {c.MAX_ASPECTS} aspects,\\n{c.MAX_TRIPLETS} triplets"];
store [label="silver/extractions.jsonl", {DATA}];
row -> hdr -> cache; cache -> hit [label="yes"]; cache -> llm [label="no"]; llm -> ok;
ok -> skip [label="yes"]; ok -> trim [label="no"]; trim -> store;
""", "TB"), "One call per review returns aspects (name, category, opinion, sentiment, evidence quote) and triplets "
          "(subject, predicate, object). Categories, predicates and sentiments are enums, so they can't drift."),

    "4. Canonicalization": (dot(f"""
raw [label="raw names from the LLM\\n(aspects + triplet ends)"];
n1 [label="brand variants\\n-> Trustpilot"]; n2 [label="NFKC + casefold"];
n3 [label="strip punctuation,\\narticles, hyphens"]; n4 [label="singularize\\nhead noun"];
norm [label="normalized names"]; sort [label="sort by frequency\\n(most frequent becomes canonical)"];
fz [label="token_sort_ratio >= {FUZZY_THRESHOLD}\\nwith an existing canonical\\nand compatible category?", {DECIDE}];
merge [label="merge into it\\n(fuzzy)"]; new [label="new canonical\\nentity"];
out [label="gold_entities +\\ngold_entity_lineage", {DATA}];
raw -> n1 -> n2 -> n3 -> n4 -> norm -> sort -> fz; fz -> merge [label="yes"]; fz -> new [label="no"];
merge -> out; new -> out;
""", "TB"), "Every step is recorded: the **Canonicalization** tab shows, per entity, which raw forms collapsed into it "
          "and whether by normalisation alone (exact) or by fuzzy match (with the score)."),

    "5. Layer 2: graph + Leiden": (dot(f"""
ents [label="canonical entities"]; rel [label="relation edges\\n(triplets, weight 2)"];
co [label="co-mention edges\\n(1/(k-1) per pair in a review)"]; g [label="weighted entity graph"];
hub [label="drop hubs ({', '.join(sorted(c.HUB_ENTITIES))})\\n+ isolated nodes"];
macro [label="Leiden macro\\nresolution {c.LEIDEN_MACRO_RESOLUTION}"];
micro [label="Leiden micro inside each macro\\nresolution {c.LEIDEN_MICRO_RESOLUTION}"];
num [label="renumber clusters by size,\\nPageRank, degree"];
ents -> rel -> g; ents -> co -> g; g -> hub -> macro -> micro -> num;
"""), "Hubs stay on the graph for display but would glue every community together, so Leiden never sees them. "
       "Micro clusters nest inside exactly one macro cluster."),

    "6. Interests & golden layer": (dot(f"""
cl [label="community"]; big [label=">= {c.MIN_INTEREST_ENTITIES} entities?", {DECIDE}];
minor [label="kept on graph,\\nno summary"]; pay [label="top {c.MAX_SUMMARY_ENTITIES} entities +\\n{c.MAX_SUMMARY_QUOTES} evidence quotes"];
cache [label="in cache?", {DECIDE}];
llm [label="{c.MODEL}\\ntitle + summary + sentiment", {LLM}];
gold [label="gold_interests\\ngold_review_interests\\ngraph.json ...", {DATA}];
cl -> big; big -> minor [label="no"]; big -> pay [label="yes"]; pay -> cache; cache -> gold [label="yes"];
cache -> llm [label="no"]; llm -> gold;
""", "TB"), "Each interest is a Leiden community named and described by the LLM from its own entities and quotes."),
}


def lineage_dot(rows, canonical: str, steps_changed: dict[str, str]) -> str:
    """raw surface forms -> normalized names -> canonical entity, edges labelled with what changed."""
    def q(s: str) -> str:
        return '"' + str(s).replace("\\", "\\\\").replace('"', '\\"') + '"'

    lines = [f'canon [label={q(canonical)}, {LLM}, penwidth=2, fontsize=13];']
    norms = {}
    for i, r in enumerate(rows.itertuples(index=False)):
        if r.normalized not in norms:
            nid = f"n{len(norms)}"
            norms[r.normalized] = nid
            lines.append(f'{nid} [label={q(r.normalized)}];')
            edge = "normalized = canonical" if r.merge == "exact" else f"fuzzy {r.fuzzy_score:.0f}"
            style = "" if r.merge == "exact" else ', color="#eb6834", fontcolor="#eb6834", style=dashed'
            lines.append(f'{nid} -> canon [label={q(edge)}{style}];')
        lines.append(f'r{i} [label={q(f"{r.raw}  ({r.n_mentions}x)")}, fillcolor="#ffffff"];')
        lines.append(f'r{i} -> {norms[r.normalized]} [label={q(steps_changed.get(r.raw, ""))}];')
    return dot("\n".join(lines))
