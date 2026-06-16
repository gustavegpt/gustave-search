"""
Gustave — Search log
====================
Appends one record per search to eval/search_log.jsonl: the query plus the full
pipeline evaluation report (constraints, decomposition, expansions that fired,
candidate/rerank verdicts with meets/caveat, the returned venues, cost, and the
human-readable report text from format_result_log).

Append-only JSONL so it's trivial to analyse later — and it's the raw material
for the self-learning loop: scan it for low-quality searches (empty results,
heavy fallback, no matches) to surface candidate learnings proactively.
"""
from __future__ import annotations

import json
import math
import os
from datetime import datetime
from pathlib import Path

LOG_PATH = Path(__file__).parent / "eval" / "search_log.jsonl"

# When set (live app, via Streamlit secret → env), records go to Supabase
# Postgres instead of the local file, which is durable across redeploys.
DB_URL_ENV = "GUSTAVE_LOG_DB_URL"

_DDL = """
create table if not exists search_log (
  id      bigserial primary key,
  ts      timestamptz default now(),
  source  text,
  query   text,
  record  jsonb
);
"""

_table_ready = False


def _san(o):
    """Make a value JSON-safe: drop NaN/inf, coerce numpy scalars to Python."""
    if isinstance(o, dict):
        return {k: _san(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_san(v) for v in o]
    if isinstance(o, float):
        return None if (math.isnan(o) or math.isinf(o)) else o
    try:
        import numpy as np
        if isinstance(o, np.floating):
            f = float(o)
            return None if (math.isnan(f) or math.isinf(f)) else f
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.bool_):
            return bool(o)
    except Exception:
        pass
    return o


def log_search(query: str, debug: dict, results: list, *,
               cost_usd: float | None = None, report_text: str = "",
               used_fallback: bool | None = None, source: str = "app_v2") -> dict:
    """Append one search record. Never raises — logging must not break a search."""
    rec = {
        "ts":            datetime.now().isoformat(timespec="seconds"),
        "source":        source,
        "query":         query,
        "constraints":   debug.get("constraints", {}),
        "decomposed":    debug.get("decomposed", {}),
        "expansions_applied": debug.get("expansions_applied", []),
        "match_count":   debug.get("match_count"),
        "alt_count":     debug.get("alt_count"),
        "result_count":  len(results),
        "used_fallback": used_fallback,
        "cost_usd":      cost_usd,
        "results": [
            {
                "name":    r.get("name"),
                "address": r.get("address"),
                "source":  r.get("source"),
                "score":   r.get("score"),
                "meets":   r.get("meets"),
                "caveat":  r.get("caveat"),
                "reason":  r.get("llm_reason"),
            }
            for r in results
        ],
        "reranker_scores": debug.get("reranker_scores", []),
        "report":        report_text,
    }
    clean = _san(rec)
    db_url = os.environ.get(DB_URL_ENV)
    if db_url:
        _log_to_pg(clean, db_url)
    else:
        try:
            LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(LOG_PATH, "a") as f:
                f.write(json.dumps(clean, ensure_ascii=False) + "\n")
        except Exception:
            pass
    return rec


def _log_to_pg(clean: dict, db_url: str) -> None:
    """Insert one record into Supabase Postgres. Never raises — best-effort."""
    global _table_ready
    try:
        import psycopg2
        with psycopg2.connect(db_url, connect_timeout=5) as conn:
            with conn.cursor() as cur:
                if not _table_ready:
                    cur.execute(_DDL)
                    _table_ready = True
                cur.execute(
                    "insert into search_log (source, query, record) values (%s, %s, %s)",
                    (clean.get("source"), clean.get("query"), json.dumps(clean, ensure_ascii=False)),
                )
            conn.commit()
    except Exception:
        # Fall back to the local file so a DB outage never loses the record.
        try:
            LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(LOG_PATH, "a") as f:
                f.write(json.dumps(clean, ensure_ascii=False) + "\n")
        except Exception:
            pass


def load_log(limit: int | None = None) -> list[dict]:
    """Read records newest-first. limit caps how many are returned."""
    if not LOG_PATH.exists():
        return []
    rows = []
    with open(LOG_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    rows.reverse()
    return rows[:limit] if limit else rows
