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

cp "$src/search_v2.py"  "$here/engine/pipeline.py"
# The test project keeps its index in gustave/cache/, but the deploy bundles it
# under engine/data/. pipeline.py is a verbatim copy of search_v2.py, so rewrite
# the CACHE_DIR path to point at the bundled data dir.
sed 's#Path(__file__).parent / "cache"#Path(__file__).parent / "data"#' \
    "$here/engine/pipeline.py" > "$here/engine/pipeline.py.tmp" \
  && mv "$here/engine/pipeline.py.tmp" "$here/engine/pipeline.py"
cp "$src/search_log.py" "$here/search_log.py"
# Engine deps: pipeline.py imports learn_core, which reads eval/expansions.json
# (the confirmed self-learning rules). Both must ship for the import to resolve.
cp "$src/learn_core.py" "$here/learn_core.py"
cp "$src/geo_zones.py"  "$here/geo_zones.py"   # pipeline imports it (location-aware, off by default)
mkdir -p "$here/eval"
cp "$src/eval/expansions.json" "$here/eval/expansions.json"
for f in faiss_vibe faiss_cuisine faiss_occasion faiss_key_facts faiss_tags faiss_full; do
  cp "$src/cache/$f.index" "$here/engine/data/$f.index"
done
cp "$src/cache/venues_v2.pkl" "$here/engine/data/venues_v2.pkl"
echo "Synced engine/pipeline.py + learn_core.py + eval/expansions.json + engine/data/ from $src"
echo "Next: git add -A && git commit -m 'refresh data' && git push"
