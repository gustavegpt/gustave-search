#!/usr/bin/env bash
# One-command deploy for the live Gustave search app.
#
#   cd deploy && ./deploy.sh ["commit message"]
#   cd deploy && ./deploy.sh --dry-run      # sync + verify, but do NOT push
#
# It (1) syncs the engine from the test project (gustave/), (2) verifies the
# deploy app imports cleanly so a broken sync never reaches users, then
# (3) commits + pushes to GitHub — Streamlit Cloud auto-redeploys on push.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
cd "$here"

DRY=0; MSG=""
for a in "$@"; do
  case "$a" in
    --dry-run) DRY=1 ;;
    *) MSG="$a" ;;
  esac
done
[ -z "$MSG" ] && MSG="deploy: refresh engine $(date '+%Y-%m-%d %H:%M')"

echo "▶ 1/3  Syncing engine from the test project (gustave/) …"
bash "$here/scripts/refresh_data.sh"

echo "▶ 2/3  Verifying the deploy app imports cleanly …"
OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE python3 - <<'PY'
import ast
for f in ("app.py", "search_log.py", "learn_core.py", "engine/pipeline.py"):
    ast.parse(open(f).read())                 # syntax
from engine import pipeline                   # resolves `import learn_core`
import search_log, learn_core                 # noqa: F401
assert hasattr(pipeline, "search") and hasattr(pipeline, "format_result_log")
# Runtime readiness: the bundled index must be where pipeline.CACHE_DIR points,
# else the live app returns 0 results for every search.
assert pipeline.indexes_ready(), (
    f"FAISS index not found under {pipeline.CACHE_DIR} — sync is broken, refusing to deploy")
print("   ✓ imports + pipeline API + index loadable OK")
PY

if [ "$DRY" = "1" ]; then
  echo "▶ 3/3  --dry-run: not committing/pushing. Changed files:"
  git -C "$here" status --short
  exit 0
fi

echo "▶ 3/3  Commit + push → Streamlit redeploys …"
git -C "$here" add -A
if git -C "$here" diff --cached --quiet; then
  echo "   Nothing changed — live app is already up to date."
  exit 0
fi
git -C "$here" commit -m "$MSG" >/dev/null
GIT_TERMINAL_PROMPT=0 git -C "$here" push origin main 2>&1 | sed -E 's#//[^@]*@#//<token>@#'
echo "✅ Pushed. Streamlit Cloud will redeploy in ~1–2 minutes."
