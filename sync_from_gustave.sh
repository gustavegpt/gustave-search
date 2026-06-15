#!/usr/bin/env bash
# Refresh the deploy snapshot from the live project before redeploying.
# Run from the deploy/ folder:  ./sync_from_gustave.sh
#
# 1. (optional) rebuild the search cache from the current DB so cleaned links
#    and newly-enriched venues are included:
#       (cd ../gustave && python3 embed_venues.py)
# 2. copy the engine + cache into this folder
# 3. git add / commit / push  → Streamlit Cloud auto-redeploys
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
src="$here/../gustave"

cp "$src/search_v2.py" "$here/search_v2.py"
for f in faiss_vibe faiss_cuisine faiss_occasion faiss_key_facts faiss_full; do
  cp "$src/cache/$f.index" "$here/cache/$f.index"
done
cp "$src/cache/venues_v2.pkl" "$here/cache/venues_v2.pkl"
echo "Synced search_v2.py + cache/ from $src"
echo "Next: git add -A && git commit -m 'refresh' && git push"
