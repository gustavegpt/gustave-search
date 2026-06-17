"""Durable logging for the live (Streamlit Cloud) app.

Streamlit Cloud's filesystem is EPHEMERAL — anything the app writes to disk
(search log, feedback/learnings) is wiped on every redeploy or container
recycle. So live feedback was effectively write-only and lost.

cloud_log mirrors each entry to a GitHub **Gist** — durable, free, no database,
no Supabase. Set two secrets in the Streamlit dashboard (and locally in
admin/.env if you want the test app to use it too):

    GUSTAVE_LOG_GIST_ID   = <id of a gist you own>
    GUSTAVE_LOG_GH_TOKEN  = <a GitHub token with the `gist` scope>

Then `learnings.jsonl` and `searches.jsonl` accumulate inside that gist; pull
them down any time with `python3 cloud_log.py pull`. When the secrets are absent
(e.g. the local test app) it's a no-op and the normal local files are used.

Never raises — logging must not break search.
"""
from __future__ import annotations
import json
import os
import urllib.request

_GIST_ID = os.environ.get("GUSTAVE_LOG_GIST_ID")
_TOKEN = os.environ.get("GUSTAVE_LOG_GH_TOKEN")
_API = "https://api.github.com/gists"


def enabled() -> bool:
    return bool(_GIST_ID and _TOKEN)


def _call(method: str, url: str, body: dict | None = None) -> dict:
    req = urllib.request.Request(url, method=method, headers={
        "Authorization": f"token {_TOKEN}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "gustave-cloud-log",
    })
    if body is not None:
        req.data = json.dumps(body).encode()
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode())


def append(filename: str, entry: dict) -> bool:
    """Append one JSONL line to <filename> inside the gist. Best-effort."""
    if not enabled():
        return False
    try:
        gist = _call("GET", f"{_API}/{_GIST_ID}")
        current = gist.get("files", {}).get(filename, {}).get("content", "") or ""
        line = json.dumps(entry, ensure_ascii=False)
        updated = current + line + "\n"
        _call("PATCH", f"{_API}/{_GIST_ID}", {"files": {filename: {"content": updated}}})
        return True
    except Exception:
        return False


def pull(out_dir: str = ".") -> None:
    """CLI helper: download both log files from the gist to local disk."""
    if not enabled():
        print("GUSTAVE_LOG_GIST_ID / GUSTAVE_LOG_GH_TOKEN not set.")
        return
    gist = _call("GET", f"{_API}/{_GIST_ID}")
    for name, meta in gist.get("files", {}).items():
        path = os.path.join(out_dir, name)
        with open(path, "w") as f:
            f.write(meta.get("content", ""))
        print(f"wrote {path} ({len(meta.get('content',''))} bytes)")


if __name__ == "__main__":
    import sys
    pull(sys.argv[2] if len(sys.argv) > 2 else ".") if (len(sys.argv) > 1 and sys.argv[1] == "pull") \
        else print(f"cloud logging {'ENABLED' if enabled() else 'disabled (no secrets)'}")
