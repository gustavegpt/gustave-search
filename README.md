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
