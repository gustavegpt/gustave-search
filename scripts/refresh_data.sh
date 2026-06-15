#!/usr/bin/env bash
# Refresh this deploy snapshot from the live Gustave project, then redeploy.
# Run from anywhere:  ./scripts/refresh_data.sh
#
# 1. (optional) rebuild the search cache from the current DB so cleaned links
#    and newly-enriched venues are included:
#       (cd <gustave-project>/gustave && python3 embed_venues.py)
# 2. copy the engine + data into this repo (engine/pipeline.py + engine/data/)
# 3. git add -A && git commit -m 'refresh' && git push  → Streamlit auto-redeploys
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"          # deploy repo root
src="${GUSTAVE_SRC:-$here/../gustave}"            # live project's gustave/ folder

cp "$src/search_v2.py" "$here/engine/pipeline.py"
for f in faiss_vibe faiss_cuisine faiss_occasion faiss_key_facts faiss_full; do
  cp "$src/cache/$f.index" "$here/engine/data/$f.index"
done
cp "$src/cache/venues_v2.pkl" "$here/engine/data/venues_v2.pkl"
echo "Synced engine/pipeline.py + engine/data/ from $src"
echo "Next: git add -A && git commit -m 'refresh data' && git push"
