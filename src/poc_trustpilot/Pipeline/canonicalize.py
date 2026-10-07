"""Layer 1 post-processing: normalise entity names and merge spelling variants into canonical nodes."""
import re
import unicodedata
from collections import Counter

import pandas as pd
from rapidfuzz import fuzz, process

from .clean import canonicalize_brand

FUZZY_THRESHOLD = 90
# Words ending in 's' that are not plurals.
_NOT_PLURAL = {"news", "series", "analytics", "status", "access", "process", "address", "business", "bonus",
               "focus", "virus", "sms", "always", "less", "terms", "this", "his", "is", "yes", "us", "gas", "plus"}


def _singular(word: str) -> str:
    if word in _NOT_PLURAL or len(word) <= 3:
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith(("sses", "shes", "ches", "xes")):
        return word[:-2]
    if word.endswith("s") and not word.endswith(("ss", "us", "is", "os")):
        return word[:-1]
    return word


def normalize_trace(name: str) -> list[tuple[str, str]]:
    """Every normalisation step with its result, so the UI can show how a surface form became a node name."""
    steps = [("raw LLM output", name or "")]
    s = canonicalize_brand(name or "")
    steps.append(("brand variants -> Trustpilot", s))
    s = unicodedata.normalize("NFKC", s).casefold().replace("̇", "")  # Turkish dotted-I artifact (notebook 02)
    steps.append(("unicode NFKC + casefold", s))
    s = re.sub(r"[\"'`.,;:!?()\[\]]", " ", s)
    s = re.sub(r"^(the|a|an)\s+", "", re.sub(r"\s+", " ", s).strip())
    s = " ".join(s.replace("-", " ").split())
    steps.append(("strip punctuation, articles, hyphens", s))
    words = s.split()
    if words:
        words[-1] = _singular(words[-1])  # head noun of an English noun phrase is the last word
    steps.append(("singularize head noun", " ".join(words)))
    return steps


def normalize_name(name: str) -> str:
    return normalize_trace(name)[-1][1]


def slug(name: str) -> str:
    return "e_" + re.sub(r"[^a-z0-9]+", "_", name).strip("_")


def build_canonical_map(names: list[str], categories: dict[str, str] | None = None) -> dict[str, str]:
    """Map every normalised name to a canonical one. Most frequent names become canonical first; a less frequent
    name merges into an existing canonical name when fuzzy similarity >= threshold and categories don't conflict."""
    categories = categories or {}
    counts = Counter(names)
    canon: list[str] = []
    mapping: dict[str, str] = {}
    for name, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        match = process.extractOne(name, canon, scorer=fuzz.token_sort_ratio, score_cutoff=FUZZY_THRESHOLD)
        if match:
            target = match[0]
            c1, c2 = categories.get(name, "other"), categories.get(target, "other")
            if c1 == c2 or "other" in (c1, c2):
                mapping[name] = target
                continue
        canon.append(name)
        mapping[name] = name
    return mapping


def canonicalize(extractions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Flatten extractions into (aspect mentions, triplets, entities, lineage) with canonical entity ids.

    lineage has one row per distinct (raw surface form, normalised name): how each LLM output became a node."""
    mentions, triplets = [], []
    for rid, ext in zip(extractions["review_id"], extractions["extraction"]):
        if not ext:
            continue
        for a in ext["aspects"]:
            mentions.append({"review_id": rid, "raw_name": a["aspect"], "name": normalize_name(a["aspect"]),
                             "category": a["category"], "opinion": a["opinion"], "sentiment": a["sentiment"],
                             "evidence": a["evidence"]})
        for t in ext["triplets"]:
            triplets.append({"review_id": rid, "subject_raw": t["subject"], "subject": normalize_name(t["subject"]),
                             "predicate": t["predicate"], "object_raw": t["object"],
                             "object": normalize_name(t["object"]), "sentiment": t["sentiment"]})
    mentions = pd.DataFrame(mentions)
    triplets = pd.DataFrame(triplets, columns=["review_id", "subject_raw", "subject", "predicate", "object_raw",
                                               "object", "sentiment"])
    mentions = mentions[mentions["name"] != ""]
    triplets = triplets[(triplets["subject"] != "") & (triplets["object"] != "")]

    # Category of a name = its most common category among aspect mentions ("other" for triplet-only names).
    cat_of = mentions.groupby("name")["category"].agg(lambda s: s.mode().iloc[0]).to_dict()
    all_names = list(mentions["name"]) + list(triplets["subject"]) + list(triplets["object"])
    cmap = build_canonical_map(all_names, cat_of)

    forms = pd.concat([
        mentions[["review_id", "raw_name", "name"]].assign(role="aspect"),
        triplets[["review_id", "subject_raw", "subject"]].set_axis(["review_id", "raw_name", "name"], axis=1)
        .assign(role="triplet subject"),
        triplets[["review_id", "object_raw", "object"]].set_axis(["review_id", "raw_name", "name"], axis=1)
        .assign(role="triplet object"),
    ])
    forms["raw_name"] = forms["raw_name"].str.strip()
    lineage = forms.groupby(["raw_name", "name"]).agg(
        n_mentions=("review_id", "size"), n_reviews=("review_id", "nunique"),
        roles=("role", lambda s: sorted(set(s))), example_review_id=("review_id", "first"),
    ).reset_index().rename(columns={"raw_name": "raw", "name": "normalized"})
    lineage["canonical"] = lineage["normalized"].map(cmap)
    lineage["merge"] = (lineage["normalized"] == lineage["canonical"]).map({True: "exact", False: "fuzzy"})
    # Same scorer build_canonical_map used, so this is the score that decided the merge.
    lineage["fuzzy_score"] = [100.0 if n == c else round(fuzz.token_sort_ratio(n, c), 1)
                              for n, c in zip(lineage["normalized"], lineage["canonical"])]

    mentions["canonical"] = mentions["name"].map(cmap)
    triplets["subject"] = triplets["subject"].map(cmap)
    triplets["object"] = triplets["object"].map(cmap)
    triplets = triplets[triplets["subject"] != triplets["object"]]

    aliases: dict[str, set] = {}
    for raw, name in zip(mentions["raw_name"], mentions["canonical"]):
        aliases.setdefault(name, set()).add(raw.strip().lower())
    for name, target in cmap.items():
        aliases.setdefault(target, set()).add(name)

    ent = pd.DataFrame({"name": sorted(set(cmap.values()))})
    canon_cat = mentions.groupby("canonical")["category"].agg(lambda s: s.mode().iloc[0])
    ent["category"] = ent["name"].map(canon_cat).fillna("other")
    ent["entity_id"] = ent["name"].map(slug)
    ent["aliases"] = ent["name"].map(lambda n: sorted(aliases.get(n, {n}) - {n}))

    id_of = dict(zip(ent["name"], ent["entity_id"]))
    mentions["entity_id"] = mentions["canonical"].map(id_of)
    triplets["subject_id"] = triplets["subject"].map(id_of)
    triplets["object_id"] = triplets["object"].map(id_of)
    lineage["entity_id"] = lineage["canonical"].map(id_of)
    return mentions.reset_index(drop=True), triplets.reset_index(drop=True), ent, lineage
