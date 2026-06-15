"""
Gustave Search Engine v2 — Semantic Multi-Embedding Pipeline
=============================================================
Implements the 6-step search architecture from the Gustave Master Plan.

Step 1: Hard constraint extraction  (Claude)
Step 2: Query decomposition         (Claude)
Step 3: Multi-embedding FAISS query (sentence-transformers + FAISS)
Step 4: Score intersection          (Python)
Step 5: LLM re-ranking              (Claude)
Step 6: Results output

Requires:
- ANTHROPIC_API_KEY in .env (or passed directly)
- FAISS indexes built by embed_venues.py
"""

from __future__ import annotations

import os
import json
import re
import numpy as np
import pandas as pd
import faiss
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

CACHE_DIR = Path(__file__).parent / "cache"
MODEL_NAME = "all-MiniLM-L6-v2"

# When cuisine is explicitly in the query, venues must score at least this
# fraction of the top cuisine score to enter the candidate pool at all.
# e.g. top Sri Lankan scores 0.82 → threshold = 0.82 * 0.45 = 0.37
# Andrew Edmunds scoring ~0.05 on Sri Lankan → eliminated before vibe counts.
CUISINE_GATE_RATIO = 0.45

# How many results to display, and how many candidates the LLM re-ranker scores
# in a single call. RERANK_POOL gates the maximum number of results possible
# when the LLM pipeline is ON (the re-ranker can only return venues it scored).
DEFAULT_TOP_K = 20
RERANK_POOL   = 20

# ── Cost estimation ────────────────────────────────────────────────────────
# Anthropic does NOT expose a remaining-credit/balance API. We instead estimate
# spend from the token usage returned on every response. Prices are USD per
# 1M tokens — update these if Anthropic changes pricing.
_MODEL_PRICING = {
    "claude-haiku-4-5-20251001": {"input": 1.00, "output": 5.00},
    "claude-sonnet-4-6":         {"input": 3.00, "output": 15.00},
}
_DEFAULT_PRICING = {"input": 1.00, "output": 5.00}

# Running token/cost tallies. _session_* accumulates for the life of the process;
# _last_search_* is reset at the start of every search() call.
_session_usage     = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
_last_search_usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}


def _record_usage(model: str, usage) -> None:
    """Accumulate token usage + estimated USD cost from one Claude response."""
    price = _MODEL_PRICING.get(model, _DEFAULT_PRICING)
    it = int(getattr(usage, "input_tokens", 0) or 0)
    ot = int(getattr(usage, "output_tokens", 0) or 0)
    cost = it / 1_000_000 * price["input"] + ot / 1_000_000 * price["output"]
    for bucket in (_session_usage, _last_search_usage):
        bucket["calls"]         += 1
        bucket["input_tokens"]  += it
        bucket["output_tokens"] += ot
        bucket["cost_usd"]      += cost


def _reset_last_search_usage() -> None:
    _last_search_usage.update(calls=0, input_tokens=0, output_tokens=0, cost_usd=0.0)


def get_last_search_usage() -> dict:
    """Token/cost estimate for the most recent search() call."""
    return dict(_last_search_usage)


def get_session_usage() -> dict:
    """Token/cost estimate accumulated since the process started."""
    return dict(_session_usage)


# Lazy-loaded singletons — loaded once, reused across searches
_model = None
_indexes: dict = {}
_df: pd.DataFrame | None = None


# ─────────────────────────────────────────────
# RESOURCE LOADING
# ─────────────────────────────────────────────
def _load_resources():
    global _model, _indexes, _df

    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)

    if not _indexes:
        for dim in ["vibe", "cuisine", "occasion", "key_facts", "full"]:
            path = CACHE_DIR / f"faiss_{dim}.index"
            if path.exists():
                _indexes[dim] = faiss.read_index(str(path))

    if _df is None:
        pkl = CACHE_DIR / "venues_v2.pkl"
        if pkl.exists():
            _df = pd.read_pickle(str(pkl))

    return _model, _indexes, _df


def indexes_ready() -> bool:
    # Core dims required for search to work. key_facts is loaded if present
    # but isn't a hard requirement — older indexes built before the 4-field
    # enricher rolled out won't have it.
    return all((CACHE_DIR / f"faiss_{dim}.index").exists()
               for dim in ["vibe", "cuisine", "occasion", "full"])


# ─────────────────────────────────────────────
# CLAUDE API HELPER
# ─────────────────────────────────────────────
_last_claude_error: str = ""


_last_claude_raw: str = ""   # stores the raw response for debugging

# Set to True by decompose_query() when it had to fall through to the
# keyword-based _fallback_decompose() — either because _claude() failed
# or because Claude returned non-JSON. Surfaced in the inspector.
_last_used_fallback: bool = False


def _strip_markdown_json(text: str) -> str:
    """Strip markdown code fences if Claude wraps JSON in them."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json or ```) and last line (```)
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    return text


def _claude(prompt: str, system: str, max_tokens: int = 512, temperature: float = 1.0) -> str:
    """Single Claude API call. Returns empty string on failure.

    temperature defaults to 1.0 (Anthropic SDK default) so the re-ranker keeps
    its existing sampling behaviour. Deterministic callers (constraint
    extraction, query decomposition) pass temperature=0.0 explicitly.
    """
    global _last_claude_error, _last_claude_raw
    _last_claude_error = ""
    _last_claude_raw = ""
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        _last_claude_error = "No API key found — check your .env file or sidebar input."
        return ""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        try:
            _record_usage(resp.model, resp.usage)
        except Exception:
            pass  # never let cost-metering break a search
        raw = resp.content[0].text.strip()
        _last_claude_raw = raw
        return _strip_markdown_json(raw)
    except Exception as e:
        _last_claude_error = str(e)
        return ""


# ─────────────────────────────────────────────
# STEP 1 — HARD CONSTRAINT EXTRACTION
# ─────────────────────────────────────────────
_CONSTRAINT_SYSTEM = """Extract ONLY hard binary constraints from the query.
Return valid JSON with these keys:
  dietary: list of strings — options: vegan, vegetarian, halal, gluten-free, kosher
  price_max: string or null — options: budget, ££, £££
  open_on: string or null — day name e.g. "Monday"
  meal_service: string or null — options: breakfast, brunch, lunch, dinner, drinks

Rules:
- ONLY extract explicit constraints. Never infer or assume.
- price_max: ONLY set if the user uses explicit price language. Examples:
    "cheap", "budget", "inexpensive" → "budget"
    "not too expensive", "mid-range", "reasonable" → "££"
    "splurge", "expensive", "fine dining", "blowout", "treat ourselves" → "£££"
  Words like "special", "nice", "lovely", "intimate", "for a date", "romantic" do NOT imply any price. Leave price_max null.
- meal_service: extract if the user mentions a meal occasion or time of day.
  "dinner", "supper", "evening meal" → dinner
  "lunch", "midday" → lunch
  "breakfast", "morning" → breakfast
  "brunch" → brunch
  "drinks", "cocktails", "bar" → drinks
- Do NOT extract cuisine, location, or vibe.
- Return ONLY valid JSON. No explanation."""


def extract_constraints(query: str) -> dict:
    defaults = {"dietary": [], "price_max": None, "open_on": None, "meal_service": None}
    raw = _claude(query, _CONSTRAINT_SYSTEM, temperature=0.0)
    if not raw:
        return defaults
    try:
        parsed = json.loads(raw)
        return {**defaults, **parsed}
    except Exception:
        return defaults


# ─────────────────────────────────────────────
# STEP 2 — QUERY DECOMPOSITION
# Location intentionally excluded — handled separately in future
# ─────────────────────────────────────────────
_DECOMPOSE_SYSTEM = """Decompose the restaurant search query into semantic dimensions.
Return valid JSON with ONLY these six keys:
  vibe_query:      short string describing mood/atmosphere/setting, or null
  cuisine_query:   short string describing food type or dishes, or null
  occasion_query:  short string describing occasion or group type, or null
  location_query:  short string with neighbourhood, area, or landmark, or null
  key_facts_query: short string with proper nouns from the query — chef names, owner names, sister/parent venue references, awards (e.g. "Michelin"), TV credits (e.g. "Bake Off"), or null
  cost_query:      short string with ANY cost/price stipulation from the query — tier words, explicit budgets, value-for-money cues, or null

Rules:
- Only include a key if clearly present in the query. Set absent keys to null.
- Keep values SHORT — 2 to 5 words max.
- cuisine_query: extract ANY food/cuisine type — Italian, Japanese, Indian, pizza, sushi, etc.
- vibe_query: atmosphere words only — romantic, intimate, lively, classy, casual, etc. Do NOT include cost language here.
- occasion_query: who/why — date night, business dinner, birthday, group, solo, etc. Do NOT include cost language here.
- location_query: area name only — Soho, Mayfair, central London, Shoreditch, etc.
- key_facts_query: only when the query references named people/venues/awards/credits — examples: "Jeremy King restaurants", "ex-River Cafe chef", "Barrafina sister", "Michelin-starred", "Bake Off judge". Leave null otherwise.
- cost_query: ALWAYS extract cost stipulations as a SEPARATE field. Examples that should populate this field:
    "cheap", "budget", "inexpensive", "affordable", "good value" → "cheap budget good value"
    "mid-range", "not too expensive", "reasonable" → "mid-range reasonable"
    "splurge", "expensive", "fine dining", "blowout", "treat ourselves", "no budget" → "splurge fine dining blowout"
    "under £30", "around £50pp", "tasting menu under £100" → quote the budget verbatim
  If the query has no cost language, set cost_query to null. Do NOT leak cost language into vibe_query or occasion_query.
- Do NOT include dietary requirements.
- Return ONLY valid JSON. No explanation."""

# Simple cuisine keyword fallback (used when Claude API is unavailable)
_CUISINE_KEYWORDS = [
    "italian", "japanese", "thai", "indian", "chinese", "french", "mexican",
    "greek", "spanish", "middle eastern", "korean", "vietnamese", "seafood",
    "steak", "pizza", "sushi", "ramen", "curry", "tapas", "dim sum",
    "turkish", "lebanese", "persian", "moroccan", "american", "british",
    "mediterranean", "asian", "european", "vegetarian", "vegan",
]
_OCCASION_KEYWORDS = [
    "date", "birthday", "anniversary", "business", "group", "solo",
    "family", "friends", "celebration", "lunch", "dinner", "brunch",
    "breakfast", "drinks", "party", "meeting", "client",
]
_VIBE_KEYWORDS = [
    "romantic", "intimate", "cosy", "cozy", "lively", "buzzy", "quiet",
    "classy", "elegant", "casual", "relaxed", "formal", "cool", "trendy",
    "hidden", "atmospheric", "moody", "bright", "rustic", "modern",
    "traditional", "understated", "unpretentious", "special", "impressive",
]
# Cost stipulations — extracted as a separate query dimension (cost_query).
# Captures explicit tier words and value-for-money cues. Numeric budgets like
# "under £30" aren't keyword-matchable here; they only get extracted when the
# Claude path is live (the fallback is best-effort).
_COST_KEYWORDS = [
    "cheap", "budget", "inexpensive", "affordable", "good value",
    "mid-range", "midrange", "reasonable", "not too expensive",
    "splurge", "expensive", "fine dining", "blowout", "pricey",
    "treat ourselves", "treat", "high-end", "upscale",
]

# Coarse signals for key_facts in fallback mode — degradation insurance only.
# Captures the two highest-value patterns: (1) credentials/awards via a small
# regex set, (2) capitalised multi-word phrases (proper nouns — chef names,
# venue references). Misses synonyms and lowercased proper nouns; that's fine.
_KEY_FACTS_REGEX = re.compile(
    r"\b("
    r"michelin(?:[- ]starred)?"
    r"|starred"
    r"|award(?:s|ed|-winning)?"
    r"|james beard"
    r"|bake off"
    r"|ex[- ][a-z]+(?:[- ][a-z]+)?"   # "ex-Noma", "ex River Cafe"
    r"|sister(?: venue| restaurant)?"
    r"|spin[- ]?off"
    r")\b",
    re.IGNORECASE,
)
_PROPER_NOUN_REGEX = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b")


def _fallback_decompose(query: str) -> dict:
    """Keyword-based decomposition used when Claude API is unavailable."""
    q = query.lower()
    cuisine = next((kw for kw in _CUISINE_KEYWORDS if kw in q), None)
    occasion = next((kw for kw in _OCCASION_KEYWORDS if kw in q), None)
    # Cost — match all cost cues, not just the first, so "cheap and good value"
    # both flow into cost_query.
    cost_hits = [kw for kw in _COST_KEYWORDS if kw in q]
    cost = " ".join(cost_hits) if cost_hits else None
    # Remove matched words from vibe search to avoid duplication / leakage
    vibe_q = q
    for matched in [cuisine, occasion]:
        if matched:
            vibe_q = vibe_q.replace(matched, "")
    for kw in cost_hits:
        vibe_q = vibe_q.replace(kw, "")
    vibe = next((kw for kw in _VIBE_KEYWORDS if kw in vibe_q), None)
    # If nothing extracted, use full query as vibe fallback
    if not cuisine and not occasion and not vibe and not cost:
        vibe = query

    # Key facts — credential phrases + capitalised multi-word proper nouns.
    # Run on the original query so casing is preserved for the proper-noun match.
    credential_hits = [m.group(0) for m in _KEY_FACTS_REGEX.finditer(query)]
    proper_noun_hits = _PROPER_NOUN_REGEX.findall(query)
    key_facts_parts = credential_hits + proper_noun_hits
    key_facts = " ".join(key_facts_parts) if key_facts_parts else None

    return {
        "vibe_query": vibe,
        "cuisine_query": cuisine,
        "occasion_query": occasion,
        "key_facts_query": key_facts,
        "cost_query": cost,
    }


def decompose_query(query: str) -> dict:
    global _last_used_fallback
    _last_used_fallback = False
    defaults = {
        "vibe_query":      None,
        "cuisine_query":   None,
        "occasion_query":  None,
        "location_query":  None,
        "key_facts_query": None,
        "cost_query":      None,
    }
    raw = _claude(query, _DECOMPOSE_SYSTEM, temperature=0.0)
    if not raw:
        _last_used_fallback = True
        return {**defaults, **_fallback_decompose(query)}
    try:
        parsed = json.loads(raw)
        return {**defaults, **parsed}
    except Exception:
        _last_used_fallback = True
        return {**defaults, **_fallback_decompose(query)}


# ─────────────────────────────────────────────
# STEPS 3 + 4 — MULTI-EMBEDDING SEARCH + INTERSECTION
# ─────────────────────────────────────────────
def _embed(text: str, model) -> np.ndarray:
    vec = model.encode([text], convert_to_numpy=True)[0]
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.astype(np.float32).reshape(1, -1)


def _query_index(index, vec: np.ndarray, k: int = 100) -> dict:
    """Returns {venue_idx: score} for top-k results."""
    scores, indices = index.search(vec, k)
    return {int(i): float(s) for i, s in zip(indices[0], scores[0]) if i >= 0}


def _location_boost(vid: int, df: pd.DataFrame, location: str) -> float:
    """Returns a score multiplier based on address match."""
    if not location:
        return 1.0
    loc_lower = location.lower().strip()
    addr = str(df.iloc[vid].get("Address", "")).lower()
    source = str(df.iloc[vid].get("Appears on", "")).lower()
    haystack = addr + " " + source
    words = loc_lower.split()
    matched = sum(1 for w in words if w in haystack and len(w) > 2)
    if matched == len(words):
        return 1.5   # full match
    if matched > 0:
        return 1.2   # partial match
    return 0.3       # no match — penalise


# Keywords that strongly signal a venue is breakfast/brunch/lunch-only
_BREAKFAST_SIGNALS = [
    "breakfast", "fry-up", "full english", "granola", "porridge", "eggs benedict",
    "sourdough toast", "avocado toast", "brunch", "morning coffee", "flat white",
    "pastry", "croissant", "açaí", "smoothie bowl", "greasy spoon", "caff",
    "all day breakfast", "fry up", "egg on toast", "beans on toast",
]
_DINNER_SIGNALS = [
    "dinner", "evening", "supper", "tasting menu", "à la carte", "a la carte",
    "wine list", "sommelier", "night cap", "late night", "dinner service",
    "evening menu", "dinner reservation",
]
_LUNCH_SIGNALS = ["lunch", "midday", "set lunch", "lunchtime", "weekday lunch"]
_DRINKS_SIGNALS = ["cocktail", "bar", "wine bar", "negroni", "aperitivo", "nightcap"]


def _meal_service_score(review: str, requested: str) -> float:
    """
    Returns a multiplier (0.1 – 1.0) based on ratio of wanted vs opposite signals.
    Uses ratio comparison so a single mention of "dinner" in a breakfast-heavy
    review doesn't rescue an obviously wrong venue.
    """
    if not requested:
        return 1.0

    review_lower = review.lower()

    signal_map = {
        "breakfast": _BREAKFAST_SIGNALS,
        "brunch":    _BREAKFAST_SIGNALS,
        "lunch":     _LUNCH_SIGNALS,
        "dinner":    _DINNER_SIGNALS,
        "drinks":    _DRINKS_SIGNALS,
    }
    opposite_map = {
        "dinner":    _BREAKFAST_SIGNALS,
        "breakfast": _DINNER_SIGNALS,
        "brunch":    _DINNER_SIGNALS,
        "lunch":     _BREAKFAST_SIGNALS,
        "drinks":    _BREAKFAST_SIGNALS,
    }

    wanted_hits   = sum(1 for kw in signal_map.get(requested, [])  if kw in review_lower)
    opposite_hits = sum(1 for kw in opposite_map.get(requested, []) if kw in review_lower)

    # Ratio-based: opposite signals dominate → penalise
    if opposite_hits >= 2 and opposite_hits > wanted_hits * 2:
        return 0.1   # e.g. 3 breakfast hits, 1 dinner hit → breakfast venue

    if opposite_hits > wanted_hits:
        return 0.4   # e.g. 2 breakfast hits, 1 dinner hit → lean mismatch

    return 1.0


# Dietary signals — used to bias the candidate pool toward venues that can
# actually satisfy a hard dietary requirement, BEFORE the LLM re-ranker runs.
# Without this, "vegan ... chef's table" retrieves meat-centric chef's-table
# venues on occasion/vibe match and they crowd out the vegan options.
_VEG_SUPPORT_SIGNALS = [
    "vegan", "plant-based", "plant based", "plantbased", "vegetarian",
    "veggie", "meat-free", "meat free", "no meat", "fully plant",
    "entirely vegan", "vegan menu", "vegan tasting", "vegan options",
    "vegan-friendly", "vegan friendly",
]
_MEAT_SIGNALS = [
    "steak", "steakhouse", "beef", "pork", "lamb", "chicken", "duck",
    "burger", "bbq", "barbecue", "rib", "ribs", "sausage", "charcuterie",
    "carnivore", "nose-to-tail", "nose to tail", "butcher", "meat-focused",
    "dry-aged", "dry aged", "chop", "veal", "venison",
]


def _dietary_score(review: str, profiles_text: str, dietary: list) -> float:
    """
    Multiplier (0.2 – 1.0) reflecting how well a venue can satisfy a HARD
    dietary requirement. Only vegan/vegetarian are scored heuristically here
    (their support/contradiction is reliably visible in review text); other
    diets (halal, kosher, gluten-free) return 1.0 and are left to the LLM
    re-ranker, which sees the dietary intent explicitly.

    Logic for vegan/vegetarian:
      - positive veg evidence            → 1.0  (confirmed suitable)
      - meat-heavy, no veg evidence      → 0.25 (almost certainly unsuitable)
      - one meat signal, no veg evidence → 0.55
      - silent on both                   → 0.7  (unconfirmed — must not outrank
                                                 a confirmed match on format alone)
    """
    if not dietary:
        return 1.0
    diet = [d.lower() for d in dietary]
    if not any(d in ("vegan", "vegetarian") for d in diet):
        return 1.0  # halal/kosher/gluten-free → leave to the re-ranker

    haystack = f"{review} {profiles_text}".lower()
    support = sum(1 for kw in _VEG_SUPPORT_SIGNALS if kw in haystack)
    meat    = sum(1 for kw in _MEAT_SIGNALS if kw in haystack)

    if support > 0:
        return 1.0
    if meat >= 2:
        return 0.25
    if meat == 1:
        return 0.55
    return 0.7


def multi_embedding_search(
    decomposed: dict,
    constraints: dict,
    model,
    indexes: dict,
    df: pd.DataFrame,
    candidate_pool: int = 50,
) -> tuple:
    """
    Query each relevant FAISS index, intersect results.
    Returns (candidates, dim_scores_per_venue) where:
      candidates       = list of (venue_idx, combined_score) sorted descending
      dim_scores_map   = {venue_idx: {dim: score, ...}} for debug
    """
    dim_map = {
        "vibe_query":      "vibe",
        "cuisine_query":   "cuisine",
        "occasion_query":  "occasion",
        "key_facts_query": "key_facts",
    }

    active = [
        (qk, dk)
        for qk, dk in dim_map.items()
        if decomposed.get(qk) and dk in indexes
    ]

    # Always search full-profile as catch-all (exclude location_query — not used in scoring)
    _scoring_keys = {"vibe_query", "cuisine_query", "occasion_query", "key_facts_query"}
    full_query = " ".join(v for k, v in decomposed.items() if k in _scoring_keys and v) or ""
    # Fold any hard dietary requirement into the full-profile query so the
    # candidate pool surfaces dietary-appropriate venues, not just format matches.
    _dietary = constraints.get("dietary") or []
    if _dietary:
        full_query = (full_query + " " + " ".join(_dietary)).strip()
    full_scores = (
        _query_index(indexes["full"], _embed(full_query, model))
        if "full" in indexes and full_query
        else {}
    )

    if not active:
        results = list(full_scores.items())
        results.sort(key=lambda x: x[1], reverse=True)
        dim_scores_map = {vid: {"full": s} for vid, s in results[:candidate_pool]}
        return results[:candidate_pool], dim_scores_map

    # Score each active dimension
    dim_score_sets = []
    dim_labels = []
    for query_key, dim_key in active:
        vec = _embed(decomposed[query_key], model)
        scores = _query_index(indexes[dim_key], vec)
        dim_score_sets.append(scores)
        dim_labels.append(dim_key)

    # Intersect: venue must appear in ALL active dimension results
    all_sets = [set(d.keys()) for d in dim_score_sets]
    intersected = set.intersection(*all_sets) if all_sets else set()
    if len(intersected) < 8:
        intersected = set.union(*all_sets)

    # ── Cuisine hard gate ──────────────────────────────────────────────────
    # If the user specified a cuisine, venues that score poorly on that
    # cuisine dimension are excluded entirely — before vibe/occasion get a vote.
    # This prevents a restaurant scoring high on "intimate romantic vibe"
    # from outranking actual cuisine matches just because it has a great review.
    cuisine_gate: set | None = None
    if "cuisine" in dim_labels:
        cuisine_scores_dict = dim_score_sets[dim_labels.index("cuisine")]
        if cuisine_scores_dict:
            top_cuisine_score = max(cuisine_scores_dict.values())
            threshold = top_cuisine_score * CUISINE_GATE_RATIO
            cuisine_gate = {vid for vid, s in cuisine_scores_dict.items() if s >= threshold}

    # Combine scores (location intentionally excluded for now)
    results = []
    dim_scores_map = {}

    for vid in intersected:
        # Hard gate: skip venues that don't meet the cuisine threshold
        if cuisine_gate is not None and vid not in cuisine_gate:
            continue
        per_dim = {label: d.get(vid, 0.0) for label, d in zip(dim_labels, dim_score_sets)}
        per_dim["full"] = full_scores.get(vid, 0.0)

        dim_avg = sum(d.get(vid, 0.0) for d in dim_score_sets) / len(dim_score_sets)
        full_bonus = per_dim["full"] * 0.3

        # Meal service filter — penalise venues that don't serve the right meal
        review = str(df.iloc[vid].get("Reviews", ""))
        meal_mult = _meal_service_score(review, constraints.get("meal_service") or "")

        # Dietary filter — down-weight venues that can't satisfy a hard diet so
        # confirmed vegan/veg venues rise into the re-ranker pool over format
        # matches (e.g. a meat chef's table can't outrank a vegan tasting menu).
        profiles_text = " ".join(str(df.iloc[vid].get(c, "") or "") for c in (
            "cuisine_profile_enriched", "key_facts_profile_enriched"))
        dietary_mult = _dietary_score(review, profiles_text, _dietary)

        combined = (dim_avg + full_bonus) * meal_mult * dietary_mult

        per_dim["meal_multiplier"] = round(meal_mult, 2)
        per_dim["dietary_multiplier"] = round(dietary_mult, 2)
        per_dim["combined"] = round(combined, 4)
        dim_scores_map[vid] = per_dim
        results.append((vid, combined))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:candidate_pool], dim_scores_map


# ─────────────────────────────────────────────
# STEP 5 — LLM RE-RANKER
# ─────────────────────────────────────────────
_RERANK_SYSTEM = """You are Gustave, a London restaurant recommendation engine.
Score how well each candidate restaurant matches the user's query.

For each candidate you receive profile fields from an enrichment pass:
  vibe        — atmosphere, energy, crowd
  cuisine     — what's actually served, style, signature dishes
  occasion    — what kinds of visits this place suits
  key_facts   — chef/owner, sister/parent venues, awards, hours, dietary, any price hints

You also receive the PARSED INTENT — how the system already decomposed the
query into dimensions and hard constraints. Treat the parsed intent as the
source of truth for what the user wants; use the raw query only for nuance.

SCOPE — SCORE ONLY THE DIMENSIONS THE USER SPECIFIED:
- The PARSED INTENT lists exactly which dimensions the user asked for (e.g.
  cuisine, vibe, occasion, key_facts, cost, dietary). Score the candidate ONLY
  against those listed dimensions.
- Any dimension NOT in the parsed intent is UNSPECIFIED. Treat it as a non-issue:
  never reward it, never penalise it, and never mention it in your reason.
- CRITICAL: the candidate profiles contain their OWN vibe/occasion/etc. Do NOT
  treat the venue's profile content as if the user requested it. A venue being
  "upscale" or "special occasion" is only relevant IF the query asked for that.
- If only CUISINE is specified, then EVERY venue that genuinely serves that
  cuisine well scores 8-10 — rank them by how central and authentic the cuisine
  match is, NOT by how fancy/casual/expensive they are. Do not down-rank a
  casual or cheap venue for not being "special occasion"; the user didn't ask.

SCORING RUBRIC (0-10) — applied ACROSS THE SPECIFIED DIMENSIONS ONLY:
  9-10  Strongly matches every specified dimension.
  7-8   Matches every specified dimension, but one is slightly soft.
  5-6   Matches the primary specified dimension (usually cuisine) but clearly
        misses ANOTHER specified dimension.
  3-4   Only one specified dimension matches when several were specified; OR the
        single specified dimension is only a loose / peripheral match.
  0-2   Fails the specified dimensions, wrong cuisine, irrelevant, OR violates a
        hard constraint (see below).

HARD RULES — these OVERRIDE the bands above:
- DIETARY IS THE #1 PRIORITY when the parsed intent lists one (vegan,
  vegetarian, halal, kosher, gluten-free). It outranks EVERYTHING — vibe,
  occasion, and format such as "tasting menu" or "chef's table". The dining
  format is worthless if the food can't be eaten.
    · The candidate must show POSITIVE evidence it can satisfy the diet
      (profile/review names the diet, "plant-based", "vegan options", a fully
      veg menu, etc.) to score 7 or above.
    · If the profile CONTRADICTS the diet (steakhouse, "meat-focused",
      "dry-aged beef", no veg options) → cap at 2.
    · If the profile is SILENT on the diet → cap at 5. We cannot confirm
      suitability, so it must NOT outrank a confirmed match however perfectly
      it nails the occasion/format.
    · A venue that satisfies the diet but only loosely matches the format
      (e.g. vegan but not strictly a tasting menu) STILL beats a perfect-format
      venue that can't confirm the diet. Rank confirmed-diet venues first.
- CUISINE IS THE SPINE. If the query names a cuisine and the candidate is a
  DIFFERENT cuisine, cap the score at 4 no matter how good the vibe or occasion.
- A price ceiling is also pass/fail: if the profile clearly contradicts it, cap
  at 2; if silent, judge on the other dimensions.
- Reviewer enthusiasm, famous chefs, or awards do NOT rescue a candidate whose
  actual requested dimensions miss.

LOCATION — TRIAL OVERRIDE:
- IGNORE any location, neighbourhood, or area in the query AND in the parsed intent.
- The candidate profiles do not include addresses — do not infer location.
- Score purely on cuisine, vibe, occasion, dietary, price, and other non-geographic dimensions.
- This override is in effect for the current trial; location filtering will be re-enabled when location settings ship.

CALIBRATION EXAMPLES (illustrate how to apply the bands):
1) Intent: cuisine=Thai, vibe=cosy casual, cost=cheap
   Candidate cuisine: "upscale Thai tasting menu, elegant, special-occasion"
   → score 4 — Thai matches but it's a fine-dining splurge, not the cheap cosy spot wanted.
2) Intent: cuisine=Italian pasta, vibe=romantic intimate
   Candidate: cuisine "handmade pasta, Roman classics", vibe "candlelit intimate date-night"
   → score 9 — Roman pasta in a candlelit intimate room hits cuisine and vibe (the only dimensions asked).
3) Intent: dietary=vegan (HARD), occasion=dinner
   Candidate key_facts: "steakhouse, dry-aged beef", no vegan options noted
   → score 2 — a steakhouse contradicts the vegan requirement.
4) Intent: cuisine=Japanese  (NOTHING else specified — no vibe, no occasion)
   Candidate A: cuisine "Tokyo izakaya, yakitori, sake", casual neighbourhood spot
   Candidate B: cuisine "Kyoto kaiseki, refined tasting menu", upscale fine dining
   → BOTH score 9 — only cuisine was asked, and both are authentically Japanese.
     Do NOT rank B above A for being upscale; "special occasion" was never requested.
     A clearly Japanese-FUSION venue (e.g. Japanese-Mediterranean robata) would score
     6-7 because the cuisine match is less central. A Korean venue would score 0-2.

For each candidate write a SHORT one-sentence reason (max 25 words) grounded in
the profile fields. Cite ONLY dimensions that appear in the parsed intent — never
claim a match or mismatch on a dimension the user did not ask for. Do NOT mention
location.

Return ONLY a valid JSON array, sorted by score descending, using the candidate
id number you were given (NOT the name):
[{"id": 3, "score": 9, "reason": "..."}, {"id": 1, "score": 6, "reason": "..."}]
No prose outside the JSON."""


def _format_intent(query: str, constraints: dict | None, decomposed: dict | None) -> str:
    """Render the parsed query understanding for the re-ranker prompt.

    Gives the LLM the SAME intent the rest of the pipeline parsed (Steps 1 & 2)
    instead of forcing it to re-derive everything from the raw query string.
    location_query is intentionally omitted — the re-ranker ignores location
    during the current trial override.
    """
    lines = [f"Query: {query}"]
    decomposed = decomposed or {}
    constraints = constraints or {}

    intent: list[str] = []
    specified: list[str] = []
    for key, label, short in [
        ("cuisine_query",   "cuisine wanted", "cuisine"),
        ("vibe_query",      "vibe wanted",    "vibe"),
        ("occasion_query",  "occasion",       "occasion"),
        ("key_facts_query", "key facts",      "key_facts"),
        ("cost_query",      "cost / budget",  "cost"),
    ]:
        val = decomposed.get(key)
        if val:
            intent.append(f"  - {label}: {val}")
            specified.append(short)

    dietary = constraints.get("dietary") or []
    if dietary:
        intent.append(f"  - dietary (HARD): {', '.join(dietary)}")
        specified.append("dietary")
    if constraints.get("price_max"):
        intent.append(f"  - price ceiling (HARD): {constraints['price_max']}")
        specified.append("price")
    if constraints.get("meal_service"):
        intent.append(f"  - meal service: {constraints['meal_service']}")
        specified.append("meal_service")

    if intent:
        lines.append("\nParsed intent (source of truth for what the user wants):")
        lines.extend(intent)
        lines.append(
            f"\nSCORE ONLY THESE DIMENSIONS: {', '.join(specified)}. "
            "Every other dimension is UNSPECIFIED — do not reward, penalise, or "
            "mention it (e.g. do NOT factor in a venue's vibe or occasion if it "
            "is not listed above)."
        )
    return "\n".join(lines)


def llm_rerank(
    query: str,
    candidates: list,
    df: pd.DataFrame,
    constraints: dict | None = None,
    decomposed: dict | None = None,
) -> tuple:
    """
    Re-rank top candidates with Claude.

    Candidates are presented with a stable integer id [1..N]; the LLM returns
    those ids (not names), so a slightly reworded name can no longer cause a
    venue to be silently scored 0 and dropped. temperature=0 makes the ordering
    deterministic. The parsed Step 1/2 intent is injected via _format_intent so
    the re-ranker scores against the same understanding as the rest of the pipeline.

    Returns (reranked_list, reranker_scores) where reranker_scores is a list
    of {name, embedding_rank, llm_score, llm_reason} dicts for the debug panel.
    """
    if not candidates:
        return [], []

    pool = candidates[:RERANK_POOL]

    candidate_lines = []
    for cid, (vid, _) in enumerate(pool, 1):
        row = df.iloc[vid]
        name = str(row.get("Restaurant", "")).strip("'\"")
        vibe   = str(row.get("vibe_profile_enriched", "")     or "").strip()
        cui    = str(row.get("cuisine_profile_enriched", "")  or "").strip()
        occ    = str(row.get("occasion_profile_enriched", "") or "").strip()
        facts  = str(row.get("key_facts_profile_enriched", "")or "").strip()
        if not any([vibe, cui, occ, facts]):
            snippet = str(row.get("Reviews", ""))[:250]
            candidate_lines.append(f"[{cid}] {name}\n    {snippet}")
        else:
            candidate_lines.append(
                f"[{cid}] {name}\n"
                f"    vibe: {vibe}\n"
                f"    cuisine: {cui}\n"
                f"    occasion: {occ}\n"
                f"    key_facts: {facts}"
            )

    intent_block = _format_intent(query, constraints, decomposed)
    prompt = f"{intent_block}\n\nCandidates:\n" + "\n\n".join(candidate_lines)
    # ~120 output tokens per candidate (score + 25-word reason) plus headroom.
    rerank_tokens = min(4000, 400 + 130 * len(pool))
    raw = _claude(prompt, _RERANK_SYSTEM, max_tokens=rerank_tokens, temperature=0.0)

    if not raw:
        return [(vid, 0.0, "") for vid, _ in pool], []

    try:
        ranked = json.loads(raw)
        id_to_info: dict[int, tuple] = {}
        for r in ranked:
            try:
                rid = int(r["id"])
            except (KeyError, ValueError, TypeError):
                continue
            id_to_info[rid] = (
                float(r.get("score", 0.0)),
                str(r.get("reason", "")).strip(),
            )

        reranked = []
        reranker_scores = []
        for cid, (vid, embed_score) in enumerate(pool, 1):
            name = str(df.iloc[vid].get("Restaurant", "")).strip("'\"")
            llm_score, llm_reason = id_to_info.get(cid, (0.0, ""))
            reranked.append((vid, llm_score, llm_reason))
            reranker_scores.append({
                "name": name,
                "embedding_rank": cid,
                "embedding_score": round(embed_score, 4),
                "llm_score": llm_score,
                "llm_reason": llm_reason,
            })

        reranked.sort(key=lambda x: x[1], reverse=True)
        reranker_scores.sort(key=lambda x: x["llm_score"], reverse=True)
        return reranked[:RERANK_POOL], reranker_scores
    except Exception:
        return [(vid, 0.0, "") for vid, _ in pool], []


# ─────────────────────────────────────────────
# STEP 6 — RESULTS FORMATTING
# ─────────────────────────────────────────────
def format_results(final: list, df: pd.DataFrame) -> list:
    results = []
    for item in final:
        # Accept 2-tuples (no LLM rerank — use_llm=False)
        # or 3-tuples (with rerank reason)
        if len(item) == 3:
            vid, score, reason = item
        else:
            vid, score = item
            reason = ""
        row = df.iloc[vid]
        review = str(row.get("Reviews", ""))
        results.append({
            "name": str(row.get("Restaurant", "")).strip("'\""),
            "address": str(row.get("Address", "")),
            "source": str(row.get("Appears on", "")),
            "review_snippet": review[:400] + "…" if len(review) > 400 else review,
            "full_review": review,
            "instagram": str(row.get("Instagram", "")),
            "ig_post": str(row.get("Ig_post", "")),
            "reservation": str(row.get("Reservation", "")),
            "menu": str(row.get("Menu", "")),
            "website": str(row.get("Website", "")),
            "rating": row.get("Rating"),
            "rating_count": row.get("Rating_count"),
            "latitude": row.get("Latitude"),
            "longitude": row.get("Longitude"),
            "score": round(score, 3),
            "llm_reason": reason,
        })
    return results


# ─────────────────────────────────────────────
# MAIN SEARCH FUNCTION
# ─────────────────────────────────────────────
def search(query: str, top_k: int = DEFAULT_TOP_K, use_llm: bool = True) -> tuple:
    """
    Full 6-step Gustave search pipeline.
    Returns (results: list, debug: dict)
    """
    if not indexes_ready():
        return [], {"error": "Indexes not built. Run: python embed_venues.py"}

    _reset_last_search_usage()
    model, indexes, df = _load_resources()
    debug = {}

    # Step 1
    constraints = extract_constraints(query)
    debug["constraints"] = constraints

    # Step 2
    decomposed = decompose_query(query)
    debug["decomposed"] = decomposed

    # Steps 3 + 4
    candidates, dim_scores_map = multi_embedding_search(decomposed, constraints, model, indexes, df)
    debug["candidate_pool"] = [
        {
            "name": str(df.iloc[vid].get("Restaurant", "")).strip("'\""),
            "address": str(df.iloc[vid].get("Address", ""))[:60],
            **dim_scores_map.get(vid, {}),
        }
        for vid, _ in candidates[:50]
    ]

    # Step 5 — LLM rerank top RERANK_POOL candidates against enriched profiles
    if use_llm and len(candidates) > 1:
        reranked, rerank_debug = llm_rerank(query, candidates[:RERANK_POOL], df, constraints, decomposed)
        final = reranked[:top_k] if reranked else candidates[:top_k]
        debug["reranker_scores"] = rerank_debug
    else:
        final = candidates[:top_k]
        debug["reranker_scores"] = []

    debug["final_count"] = len(final)

    # Step 6
    results = format_results(final[:top_k], df)
    return results, debug


# ─────────────────────────────────────────────
# RESULT LOG (for copy-paste evaluation)
# ─────────────────────────────────────────────
def format_result_log(query: str, debug: dict, results: list) -> str:
    lines = []
    lines.append(f"QUERY: {query}")
    lines.append("")

    c = debug.get("constraints", {})
    lines.append("STEP 1 — Constraints:")
    lines.append(f"  meal_service : {c.get('meal_service')}")
    lines.append(f"  dietary      : {c.get('dietary')}")
    lines.append(f"  price_max    : {c.get('price_max')}")
    lines.append(f"  open_on      : {c.get('open_on')}")
    lines.append("")

    d = debug.get("decomposed", {})
    lines.append("STEP 2 — Decomposed:")
    lines.append(f"  vibe_query    : {d.get('vibe_query')}")
    lines.append(f"  cuisine_query : {d.get('cuisine_query')}")
    lines.append(f"  occasion_query: {d.get('occasion_query')}")
    lines.append(f"  key_facts_query: {d.get('key_facts_query')}")
    lines.append(f"  cost_query    : {d.get('cost_query')} (extracted only — not yet wired into scoring)")
    lines.append(f"  location_query: {d.get('location_query')} (extracted only — not used in scoring)")
    lines.append(f"  claude_error  : {_last_claude_error or 'none'}")
    lines.append(f"  used_fallback : {_last_used_fallback}")
    lines.append("")

    lines.append(f"STEP 3/4 — Candidates before re-rank: {debug.get('candidates', len(debug.get('candidate_pool', [])))}")
    lines.append("")

    lines.append("TOP 50 CANDIDATES:")
    for i, c in enumerate(debug.get("candidate_pool", []), 1):
        meal = c.get("meal_multiplier", 1.0)
        combined = c.get("combined", 0)
        lines.append(f"  {i:2}. {c.get('name','')} | meal×={meal} | score={combined}")
    lines.append("")

    lines.append("FINAL RESULTS:")
    for i, r in enumerate(results, 1):
        lines.append(f"  {i}. {r['name']} (score: {r['score']})")
        lines.append(f"     {r['address'][:70]}")
        if r.get("llm_reason"):
            lines.append(f"     reason: {r['llm_reason']}")

    return "\n".join(lines)
