"""Paths, model settings and tunables for the interest-extraction pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RAW_CSV = ROOT / "Data_Advisory_Random_Sample_tp_2021 (3) (1).csv"

DATA = ROOT / "data"
BRONZE = DATA / "bronze"
SILVER = DATA / "silver"
GOLD = DATA / "gold"

# LLM - cheapest viable setup: Sonnet 5.5, low effort, thinking off.
MODEL = "claude-sonnet-5-5"
EFFORT = "low"
MAX_WORKERS = 8
PROMPT_VERSION = "v2"  # bump to invalidate cached extractions after a prompt/schema change

# Token caps. Reviews are short (sampled median ~66 tokens, max ~400), so these rarely bind.
MAX_REVIEW_CHARS = 1500           # ~400 tokens; longer review text is cut at a word boundary
MAX_ASPECTS, MAX_TRIPLETS = 4, 3  # per review - asked for in the prompt, enforced after parsing
EVIDENCE_MAX_WORDS = 12
MAX_OUTPUT_TOKENS_EXTRACTION = 1024  # a capped extraction is ~250-400 tokens; this is a runaway guard, not a budget
MAX_OUTPUT_TOKENS_SUMMARY = 512
MAX_SUMMARY_ENTITIES, MAX_SUMMARY_QUOTES, MAX_QUOTE_CHARS = 15, 8, 160

# USD per million tokens for MODEL - used only for the cost printout.
PRICE_INPUT, PRICE_OUTPUT = 2.00, 10.00
PRICE_CACHE_WRITE, PRICE_CACHE_READ = 2.50, 0.20

SAMPLE_SIZE = 50
SEED = 42

# Leiden: macro communities on the whole graph, micro communities inside each macro one.
LEIDEN_MACRO_RESOLUTION = 0.6
LEIDEN_MICRO_RESOLUTION = 1.5
MIN_INTEREST_ENTITIES = 3  # smaller communities are kept on the graph but not summarised as interests

# Generic nodes that appear in most reviews (the brand, and the placeholder roles the prompt asks for).
# They stay on the graph but are excluded from clustering so they don't glue every community together.
HUB_ENTITIES = {"trustpilot", "reviewer", "business"}

SOURCE_GROUP = {
    "invitationlinkapi": "invited", "invitationapi": "invited", "afsv2": "invited", "basiclink": "invited",
    "organic": "organic",
    "domainlink": "other",
}
