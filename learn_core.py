"""
Gustave — Learning core (registry + expansion layer)
=====================================================
The import-light half of the self-learning loop. No pandas / no model load, so
search_v2.py can call it on every query with zero overhead.

Two JSON files live in eval/ next to the gold set:

  learnings.json   — the raw ledger. Every piece of feedback (from the inspector
                     "fix this" box or a manual edit) lands here untouched, with
                     a status the triage brain (learn.py) updates.

  expansions.json  — the ACTIVE rules the pipeline consumes at decompose time.
                     A rule only reaches this file after learn.py has eval-gated
                     it (Hit@10 improved, nothing regressed). This is what makes
                     a learning actually change search behaviour.

Rule shape:
  {
    "id": "exp-2026-06-15-001",
    "field": "cuisine_query",          # which decomposed field to enrich
    "trigger": ["meat", "meat tasting"],   # substrings in the raw query that fire it
    "add": ["barbecue", "smokehouse", "grill", "steakhouse"],  # venue vocabulary
    "source_learning": "2026-06-15-001",
    "eval": {"before_hit10": 0.70, "after_hit10": 0.73}
  }
"""
from __future__ import annotations

import json
from pathlib import Path

EVAL_DIR        = Path(__file__).parent / "eval"
LEARNINGS_PATH  = EVAL_DIR / "learnings.json"
EXPANSIONS_PATH = EVAL_DIR / "expansions.json"

EXPANDABLE_FIELDS = {"cuisine_query", "vibe_query", "occasion_query", "key_facts_query"}


# ── load / save ───────────────────────────────────────────────────────────────

def _load(path: Path, default: dict) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def load_expansions() -> dict:
    return _load(EXPANSIONS_PATH, {"rules": []})


def save_expansions(data: dict) -> None:
    EVAL_DIR.mkdir(exist_ok=True)
    with open(EXPANSIONS_PATH, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_learnings() -> dict:
    return _load(LEARNINGS_PATH, {"learnings": []})


def save_learnings(data: dict) -> None:
    EVAL_DIR.mkdir(exist_ok=True)
    with open(LEARNINGS_PATH, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def append_learning(entry: dict) -> dict:
    """Append a raw feedback entry to the ledger; returns the stored entry."""
    data = load_learnings()
    data.setdefault("learnings", []).append(entry)
    save_learnings(data)
    return entry


# ── the hot path: apply expansions at decompose time ────────────────────────────

def apply_expansions(query: str, decomposed: dict, expansions: dict | None = None) -> tuple[dict, list]:
    """
    Enrich the decomposed query with confirmed expansion vocabulary.

    For each rule whose trigger appears in the raw query, append the rule's
    `add` terms to the target decomposed field (deduped, order-preserving).
    Returns (new_decomposed, applied) where `applied` records what fired so the
    inspector can show it. Pure + side-effect free — safe on every search.
    """
    if expansions is None:
        expansions = load_expansions()

    q = (query or "").lower()
    out = dict(decomposed)
    applied: list = []

    for rule in expansions.get("rules", []):
        field = rule.get("field", "cuisine_query")
        if field not in EXPANDABLE_FIELDS:
            continue
        fired_triggers = [t for t in rule.get("trigger", []) if t and t.lower() in q]
        if not fired_triggers:
            continue
        add_terms = [t for t in rule.get("add", []) if t]
        if not add_terms:
            continue

        existing = (out.get(field) or "").strip()
        existing_words = existing.lower().split()
        new_terms = [t for t in add_terms if t.lower() not in existing_words]
        if not new_terms:
            continue
        out[field] = (existing + " " + " ".join(new_terms)).strip() if existing else " ".join(new_terms)
        applied.append({
            "id": rule.get("id"),
            "field": field,
            "added": new_terms,
            "triggered_by": fired_triggers,
            "source_learning": rule.get("source_learning"),
        })

    return out, applied
