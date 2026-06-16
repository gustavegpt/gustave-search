# Gustave — search (test build)

A self-contained Streamlit app hosting the Gustave London restaurant search for
testers, behind a passcode and a spend cap. It reuses the real search pipeline
over a bundled snapshot of the search index.

**No secrets live in this repo.** The Anthropic API key and passcode are set in
the Streamlit Cloud dashboard, never committed.

```
.
├── app.py                  # Streamlit entry point (the UI)
├── requirements.txt
├── runtime.txt             # python-3.11
├── README.md
├── .streamlit/
│   ├── config.toml         # theme
│   └── secrets.toml.example
├── engine/                 # the search engine package
│   ├── __init__.py
│   ├── pipeline.py         # the 6-step search pipeline
│   └── data/               # bundled search index
│       ├── faiss_*.index   # 5 semantic indexes
│       └── venues_v2.pkl   # venue data (names, reviews, links, ratings, geo)
└── scripts/
    └── refresh_data.sh     # re-sync engine + data from the main project
```

---

## Deploy on Streamlit Community Cloud

1. **[share.streamlit.io](https://share.streamlit.io)** → sign in with the GitHub
   account that owns this repo → **Create app**.
2. Repository **`gustavegpt/gustave-search`**, branch **`main`**, main file **`app.py`**.
3. **Advanced settings → Secrets** — paste (your real values):

   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   GUSTAVE_PASSCODE  = "pick-a-word"
   GUSTAVE_SESSION_CAP = "25"
   GUSTAVE_DAILY_CAP   = "300"
   ```

4. **Deploy.** First build takes a few minutes (installs torch + downloads the
   embedding model). Share the resulting URL **and** the passcode with testers.

---

## Cost & limits

- Each search runs Claude on your key (~a few cents). `GUSTAVE_SESSION_CAP`
  (default 25) caps one visitor per session; `GUSTAVE_DAILY_CAP` (default 300)
  is a best-effort daily ceiling across everyone. Adjust both in Secrets — no
  redeploy needed.

## Search log (durable, via Supabase)

Every live search — the query plus its full evaluation report (constraints,
decomposition, candidate pool, re-ranker verdicts, returned venues, cost) — can
be logged to a Postgres table so you can review what testers searched and how
the engine did. Streamlit Cloud's own disk is wiped on every redeploy, so the
log goes to **Supabase** (free tier) instead.

**One-time setup:**
1. Create a free project at [supabase.com](https://supabase.com).
2. In the Supabase SQL editor, run:
   ```sql
   create table if not exists search_log (
     id      bigserial primary key,
     ts      timestamptz default now(),
     source  text,
     query   text,
     record  jsonb
   );
   ```
   (The app also auto-creates this on first write, so this step is optional.)
3. **Project Settings → Database → Connection string → URI** (use the *Session
   pooler* string). Copy it and add to Streamlit secrets:
   ```toml
   GUSTAVE_LOG_DB_URL = "postgresql://postgres:[PASSWORD]@db.[REF].supabase.co:5432/postgres"
   ```
4. Redeploy. Searches now append to `search_log`. View them in the Supabase
   table editor, or query with SQL. If the secret is absent, logging is simply
   off (no errors). If the DB is briefly unreachable, the record falls back to a
   local file so nothing is lost mid-session.

The **test app** (`gustave/app_v2.py`) logs to a local `eval/search_log.jsonl`
with an in-app viewer + download — no Supabase needed there.

## Updating the data later

```bash
cd <gustave-project>/gustave && python3 embed_venues.py   # rebuild the index
./scripts/refresh_data.sh                                 # copy engine + data here
git add -A && git commit -m "refresh data" && git push    # auto-redeploys
```

## Run locally

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # fill in real values
python3 -m streamlit run app.py
```
