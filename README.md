# Gustave — public search (friends test build)

A self-contained Streamlit app that hosts the Gustave restaurant search for a
handful of testers. It reuses the real search pipeline (`search_v2.py`) over a
snapshot of the search index (`cache/`), behind a passcode and a spend cap.

**Nothing secret is in this folder.** The Anthropic API key and passcode are
set later in the Streamlit dashboard, never committed.

```
deploy/
├── app.py                 # the friend-facing search UI (passcode + spend cap)
├── search_v2.py           # search engine (copied from ../gustave)
├── cache/                 # search index snapshot (5 faiss_*.index + venues_v2.pkl)
├── requirements.txt
├── runtime.txt            # python-3.11
├── .streamlit/
│   ├── config.toml        # theme
│   └── secrets.toml.example
├── .gitignore             # ignores secrets.toml
└── sync_from_gustave.sh   # refresh engine+cache before a redeploy
```

---

## One-time deploy (≈10 min)

### 1. Put this folder on GitHub

You need a free GitHub account. From a terminal:

```bash
cd /Users/andreidonko/Gustave/deploy
git init -b main
git add -A
git commit -m "Gustave search — friends test build"
```

Create an **empty** repo on github.com (e.g. `gustave-search`, private is fine —
Streamlit Cloud can read private repos), then:

```bash
git remote add origin https://github.com/<your-username>/gustave-search.git
git push -u origin main
```

### 2. Deploy on Streamlit Community Cloud (free)

1. Go to **https://share.streamlit.io** and sign in with GitHub.
2. **Create app → Deploy a public app from GitHub** (works for private repos too).
3. Select your `gustave-search` repo, branch `main`, **Main file path** `app.py`.
4. Click **Advanced settings → Secrets** and paste (your real values):

   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   GUSTAVE_PASSCODE  = "pick-a-word"
   GUSTAVE_SESSION_CAP = "25"
   GUSTAVE_DAILY_CAP   = "300"
   ```

5. **Deploy.** First build takes a few minutes (it installs torch + downloads the
   ~90 MB embedding model). When it's up you'll get a URL like
   `https://gustave-search.streamlit.app`.

### 3. Share with friends

Send them the URL **and** the passcode. That's it.

---

## Cost & limits

- Every search runs the LLM pipeline on **your** Anthropic key — roughly a few
  cents per search.
- `GUSTAVE_SESSION_CAP` (default 25) limits one visitor per browser session.
- `GUSTAVE_DAILY_CAP` (default 300) is a best-effort ceiling across everyone per
  day (~a few £/day worst case). It resets daily and on app restart. Raise/lower
  both in the Secrets panel anytime — no redeploy needed.
- For a hard guarantee, create a **budget-limited Anthropic key** for this app.

---

## Updating the data or app later

The `cache/` here is a snapshot. After more scraping / enrichment / link fixes
in the main project, refresh and push:

```bash
cd /Users/andreidonko/Gustave/gustave && python3 embed_venues.py   # rebuild index
cd ../deploy && ./sync_from_gustave.sh                             # copy engine+cache
git add -A && git commit -m "refresh data" && git push             # auto-redeploys
```

Streamlit Cloud redeploys automatically on every push to `main`.

---

## Run locally first (optional)

```bash
cd deploy
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # fill in real values
python3 -m streamlit run app.py
```
