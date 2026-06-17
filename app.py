"""
Gustave — public search (friends test build)
=============================================
A lean, friend-facing version of the Gustave search UI for Streamlit
Community Cloud. Reuses the real search pipeline (engine/pipeline.py) but
hides all developer surfaces (pipeline inspector, API-key box, eval log) and adds:

  • a passcode gate (GUSTAVE_PASSCODE)
  • a per-session + per-day search cap so a shared link can't run up the
    Anthropic bill (GUSTAVE_SESSION_CAP / GUSTAVE_DAILY_CAP)

Secrets (set in the Streamlit Cloud dashboard, NOT committed):
  ANTHROPIC_API_KEY, GUSTAVE_PASSCODE, GUSTAVE_SESSION_CAP, GUSTAVE_DAILY_CAP
"""
from __future__ import annotations

import os
from datetime import date

import streamlit as st
import streamlit.components.v1 as components

# ── Bridge Streamlit secrets → env vars ───────────────────────────────────────
# The engine reads os.environ; Streamlit Cloud exposes secrets via st.secrets.
# Accessing st.secrets with no secrets file raises, so guard it (local runs).
try:
    _secrets = dict(st.secrets)
except Exception:
    _secrets = {}
for _k in ("ANTHROPIC_API_KEY", "GUSTAVE_PASSCODE",
           "GUSTAVE_SESSION_CAP", "GUSTAVE_DAILY_CAP",
           "GUSTAVE_LOG_DB_URL", "GUSTAVE_ADMIN_KEY",
           "GUSTAVE_LOG_GIST_ID", "GUSTAVE_LOG_GH_TOKEN"):
    if _k not in os.environ and _k in _secrets:
        os.environ[_k] = str(_secrets[_k])

from engine import pipeline  # noqa: E402  (after env bridge)
import search_log            # noqa: E402  (search + feedback log; local file by default)

st.set_page_config(page_title="Gustave — London restaurant search",
                   page_icon="🍽️", layout="wide")

SESSION_CAP = int(os.environ.get("GUSTAVE_SESSION_CAP", "25"))
DAILY_CAP = int(os.environ.get("GUSTAVE_DAILY_CAP", "300"))


# ── Passcode gate ─────────────────────────────────────────────────────────────
_passcode = os.environ.get("GUSTAVE_PASSCODE")
if _passcode and not st.session_state.get("_ok"):
    st.markdown("### 🔒 Gustave")
    st.caption("Private test build. Enter the passcode you were given.")
    with st.form("gate"):
        entered = st.text_input("Passcode", type="password")
        if st.form_submit_button("Enter"):
            if entered == _passcode:
                st.session_state["_ok"] = True
                st.rerun()
            else:
                st.error("Wrong passcode.")
    st.stop()


# ── Shared daily counter (process-global, best-effort) ────────────────────────
@st.cache_resource
def _daily():
    return {"day": date.today().isoformat(), "count": 0}


def _budget_state() -> tuple[bool, str]:
    """Return (allowed, message). Enforces session + daily caps."""
    d = _daily()
    today = date.today().isoformat()
    if d["day"] != today:                      # new day → reset
        d["day"], d["count"] = today, 0
    sess = st.session_state.get("_searches", 0)
    if sess >= SESSION_CAP:
        return False, (f"You've used all {SESSION_CAP} searches for this "
                       "session. Refresh later to continue — this cap keeps the "
                       "shared test bill in check.")
    if d["count"] >= DAILY_CAP:
        return False, ("Gustave has hit its shared daily search limit. "
                       "Please try again tomorrow 🙏")
    return True, ""


def _record_search() -> None:
    st.session_state["_searches"] = st.session_state.get("_searches", 0) + 1
    _daily()["count"] += 1


# ── Header ────────────────────────────────────────────────────────────────────
st.title("🍽️ Gustave")
st.caption("London restaurant search · built only on editorial reviews from "
           "named critics · no paid placements, no crowd ratings")

if not os.environ.get("ANTHROPIC_API_KEY"):
    st.error("Search is temporarily unavailable (missing API configuration). "
             "Please let Andrei know.")
    st.stop()
if not pipeline.indexes_ready():
    st.error("Search index unavailable. Please let Andrei know.")
    st.stop()

nav = st.radio("Navigation", ["🔎 Search", "🗺️ Browse all"],
               horizontal=True, label_visibility="collapsed", key="nav")
st.divider()

# ── Admin: view + download captured searches & feedback (gated by GUSTAVE_ADMIN_KEY) ──
_admin_key = os.environ.get("GUSTAVE_ADMIN_KEY")
if _admin_key:
    with st.sidebar:
        st.markdown("**🔒 Admin**")
        if st.text_input("Admin key", type="password", key="_adminkey") == _admin_key:
            _rows = search_log.load_log()
            st.caption(f"📜 {len(_rows)} searches logged")
            try:
                st.download_button("⬇ Search log (JSONL)", search_log.LOG_PATH.read_bytes(),
                                   file_name="gustave_search_log.jsonl", mime="application/json",
                                   use_container_width=True)
            except Exception:
                st.caption("No search log yet.")
            try:
                import learn_core
                _fb = learn_core.load_learnings().get("learnings", [])
                st.caption(f"💬 {len(_fb)} feedback entries")
                if _fb:
                    st.download_button("⬇ Feedback (learnings.json)",
                                       learn_core.LEARNINGS_PATH.read_bytes(),
                                       file_name="learnings.json", mime="application/json",
                                       use_container_width=True)
            except Exception:
                st.caption("No feedback yet.")
            st.caption("⚠️ Cloud storage resets on redeploy — download to keep.")


def _ig_handle(url: str) -> str:
    """Extract the @handle from an Instagram profile URL."""
    import re
    m = re.search(r"instagram\.com/([^/?#]+)", url or "")
    return m.group(1).strip("/") if m else ""


def _ig_post_embed(url: str) -> str:
    """Official Instagram post embed (real post image) from a /p/ /reel/ /tv/
    URL. Returns iframe HTML, or '' if the URL isn't a post."""
    import re
    m = re.search(r"instagram\.com/(?:p|reel|tv)/([^/?#]+)", url or "")
    if not m:
        return ""
    return (
        f"<iframe src='https://www.instagram.com/p/{m.group(1)}/embed/' "
        "width='100%' height='500' frameborder='0' scrolling='no' "
        "style='border-radius:12px;border:1px solid #e0d8c8;background:#fff;'>"
        "</iframe>"
    )


def _ig_card(url: str) -> str:
    """A branded Instagram card linking to the venue's profile (latest posts).
    Live per-venue post feeds aren't available from Instagram for arbitrary
    public profiles, so this is a styled link to the profile grid."""
    handle = _ig_handle(url)
    if not handle:
        return ""
    glyph = (
        "<svg width='18' height='18' viewBox='0 0 24 24' fill='none' "
        "stroke='white' stroke-width='2' stroke-linecap='round' "
        "stroke-linejoin='round' style='vertical-align:middle'>"
        "<rect x='2' y='2' width='20' height='20' rx='5'/>"
        "<circle cx='12' cy='12' r='4'/>"
        "<circle cx='17.5' cy='6.5' r='1.2' fill='white' stroke='none'/></svg>"
    )
    return (
        f"<a href='{url}' target='_blank' style='display:inline-flex;"
        "align-items:center;gap:8px;padding:7px 14px;border-radius:12px;"
        "text-decoration:none;color:#fff;font-weight:600;font-size:0.9rem;"
        "background:linear-gradient(45deg,#feda75,#fa7e1e,#d62976,#962fbf,#4f5bd5);'>"
        f"{glyph}<span>@{handle} · latest on Instagram →</span></a>"
    )


def _num(x):
    """Float or None — treats NaN / non-numeric as None (NaN is truthy in Python,
    which is the trap behind 'cannot convert float NaN to integer')."""
    import math
    try:
        f = float(x)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _stars(rt) -> str:
    f = _num(rt)
    if f is None:
        return ""
    n = int(round(f))
    return "★" * n + "☆" * (5 - n)


def _g(v: dict, *keys):
    """First non-empty value among keys (handles result dicts + pkl rows).
    Skips None, '', 'nan', and float NaN."""
    for k in keys:
        val = v.get(k)
        if val in (None, "", "nan"):
            continue
        if isinstance(val, float) and val != val:  # NaN
            continue
        return val
    return None


def _render_card(v: dict, is_alt: bool = False) -> None:
    """Render one venue card — used by search results and map-click popups.
    is_alt=True leads with the caveat (why it's an alternative, not a full match)."""
    st.subheader(str(_g(v, "name", "Restaurant") or "—").strip("'\""))
    addr = _g(v, "address", "Area", "Address") or "London"
    src = _g(v, "source", "Appears on") or ""
    st.caption(f"📍 {addr}" + (f"   ·   {src}" if src else ""))
    rt = _num(_g(v, "rating", "Rating"))
    if rt:
        rc = _num(_g(v, "rating_count", "Rating_count"))
        cnt = f" · {int(rc):,} Google reviews" if rc else " · Google"
        st.caption(f"{_stars(rt)}  **{rt:g}**{cnt}")
    if is_alt and v.get("caveat"):
        st.warning(f"⚠️ **Doesn't tick:** {v['caveat']}")
    if v.get("llm_reason"):
        st.markdown(f"💡 *{v['llm_reason']}*")
    if v.get("review_snippet"):
        st.markdown(f"> {v['review_snippet']}")
    links = []
    for label, keys in (("Book a table", ("reservation", "Reservation")),
                        ("Menu", ("menu", "Menu")),
                        ("Website", ("website", "Website"))):
        val = _g(v, *keys)
        if val:
            links.append(f"[{label}]({val})")
    if links:
        st.markdown("  ·  ".join(links))
    embed = _ig_post_embed(_g(v, "ig_post", "Ig_post") or "")
    if embed:
        components.html(embed, height=520)
    else:
        igp = _g(v, "instagram", "Instagram")
        if igp:
            st.markdown(_ig_card(igp), unsafe_allow_html=True)


def _venue_map(points, key: str):
    """points: DataFrame with lat, lon, name, rating_label (+ card fields).
    Renders a pydeck map (hover tooltip, click-to-select). Returns the clicked
    row dict or None."""
    import pydeck as pdk
    if points is None or points.empty:
        st.caption("No mapped locations to show.")
        return None
    layer = pdk.Layer(
        "ScatterplotLayer", data=points, get_position="[lon, lat]",
        get_fill_color="[196, 91, 60, 210]", get_radius=70,
        radius_min_pixels=5, radius_max_pixels=16, pickable=True,
        auto_highlight=True)
    deck = pdk.Deck(
        layers=[layer], map_style="light",
        initial_view_state=pdk.ViewState(
            latitude=float(points["lat"].median()),
            longitude=float(points["lon"].median()),
            zoom=11 if len(points) < 60 else 10),
        tooltip={"html": "<b>{name}</b><br/>{rating_label}",
                 "style": {"backgroundColor": "#2B2B2B", "color": "#fff",
                           "fontSize": "0.8rem"}})
    event = st.pydeck_chart(deck, on_select="rerun",
                            selection_mode="single-object", key=key)
    objs = (getattr(event, "selection", None) or {}).get("objects", {})
    for layer_objs in objs.values():
        if layer_objs:
            return layer_objs[0]
    return None


def _rating_label(rt, rc) -> str:
    rt, rc = _num(rt), _num(rc)
    if not rt:
        return ""
    return f"⭐ {rt:g}" + (f" ({int(rc):,})" if rc else "")


# ══════════════════════════════════════════════════════════════════════════════
# BROWSE ALL — map of London + alphabetical directory of every venue
# ══════════════════════════════════════════════════════════════════════════════
if nav == "🗺️ Browse all":
    @st.cache_data(show_spinner=False)
    def _all_venues():
        import pandas as pd
        df = pd.read_pickle("engine/data/venues_v2.pkl")
        df = df.copy()
        df["Restaurant"] = df["Restaurant"].astype(str).str.strip("'\"")
        for c in ("Latitude", "Longitude"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df.sort_values("Restaurant", key=lambda s: s.str.lower())

    vdf = _all_venues()
    st.subheader(f"All {len(vdf)} restaurants")

    import pandas as pd
    pts = vdf.copy()
    pts["lat"], pts["lon"] = pts["Latitude"], pts["Longitude"]
    pts = pts.dropna(subset=["lat", "lon"])
    pts = pts[(pts.lat.between(51.2, 51.8)) & (pts.lon.between(-0.6, 0.4))]
    has_r = "Rating" in pts.columns
    pts["name"] = pts["Restaurant"]
    pts["rating_label"] = [
        _rating_label(r.get("Rating") if has_r else None, r.get("Rating_count"))
        for _, r in pts.iterrows()]
    st.caption(f"📍 {len(pts)} venues across London — hover for a name, "
               "click a dot to see the restaurant.")
    clicked = _venue_map(
        pts[["lat", "lon", "name", "rating_label", "Restaurant", "Address",
             "Rating", "Rating_count", "Reservation", "Menu", "Website",
             "Instagram", "Ig_post"]] if has_r else
        pts[["lat", "lon", "name", "rating_label", "Restaurant", "Address",
             "Reservation", "Menu", "Website", "Instagram", "Ig_post"]],
        key="browse_map")
    if clicked:
        with st.container(border=True):
            _render_card(clicked)

    st.divider()
    st.caption("A–Z directory — scroll the list; click a link to open it.")
    cols = ["Restaurant", "Address"]
    if has_r:
        cols.append("Rating")
    cols += ["Reservation", "Menu", "Website"]
    table = vdf[cols].rename(columns={"Address": "Area"})
    st.dataframe(
        table, height=560, hide_index=True, use_container_width=True,
        column_config={
            "Rating": st.column_config.NumberColumn("⭐", format="%.1f"),
            "Reservation": st.column_config.LinkColumn("Book", display_text="Book →"),
            "Menu": st.column_config.LinkColumn("Menu", display_text="Menu →"),
            "Website": st.column_config.LinkColumn("Website", display_text="Site →"),
        },
    )
    st.caption("Gustave is an early test build — data and links are still being "
               "refined. Spotted something off? Tell Andrei.")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# SEARCH
# ══════════════════════════════════════════════════════════════════════════════
query = st.text_input(
    "Search", label_visibility="collapsed",
    placeholder="e.g. cosy Italian for a date in Shoreditch")
go = st.button("Search", type="primary")
st.caption("Try: *romantic Italian in Soho* · *quiet first date* · "
           "*lively group birthday* · *fine dining tasting menu*")

_left = SESSION_CAP - st.session_state.get("_searches", 0)
st.caption(f"_{max(_left, 0)} of {SESSION_CAP} searches left this session._")


# ── Run search ────────────────────────────────────────────────────────────────
if go and query.strip():
    allowed, msg = _budget_state()
    if not allowed:
        st.warning(msg)
        st.stop()
    with st.spinner("Searching London's best reviews …"):
        results, _debug = pipeline.search(query, top_k=15, use_llm=True)
    _record_search()
    # Durably log the search + its full evaluation report (→ Supabase if
    # GUSTAVE_LOG_DB_URL is set, else a local file). Never blocks a search.
    try:
        _report = pipeline.format_result_log(query, _debug, results)
        search_log.log_search(query, _debug, results, report_text=_report, source="live")
        import cloud_log  # durable mirror to gist (survives redeploys)
        cloud_log.append("searches.jsonl", {
            "query": query,
            "matches": _debug.get("match_count"), "alts": _debug.get("alt_count"),
            "decomposed": _debug.get("decomposed"),
            "results": [r.get("name") for r in results],
        })
    except Exception:
        pass
    # Persist so a map-marker click (which reruns) keeps the results.
    st.session_state["_results"] = results
    st.session_state["_query"] = query
    st.session_state.pop("search_map", None)  # clear stale map selection
elif go:
    st.warning("Please type something to search for.")

results = st.session_state.get("_results")
qy = st.session_state.get("_query", "")
if results:
    matches = [r for r in results if not r.get("is_more") and not r.get("is_alt", not r.get("meets", True))]
    alternatives = [r for r in results if not r.get("is_more") and r.get("is_alt", not r.get("meets", True))]
    more = [r for r in results if r.get("is_more")]

    # Map of the results — hover for a name, click a dot to open its card.
    import pandas as pd
    rows = []
    for r in results:
        try:
            lat, lon = float(r.get("latitude")), float(r.get("longitude"))
        except (TypeError, ValueError):
            continue
        rows.append({
            "lat": lat, "lon": lon, "name": r["name"],
            "rating_label": _rating_label(r.get("rating"), r.get("rating_count")),
            "address": r.get("address", ""), "source": r.get("source", ""),
            "rating": r.get("rating"), "rating_count": r.get("rating_count"),
            "reservation": r.get("reservation", ""), "menu": r.get("menu", ""),
            "website": r.get("website", ""), "instagram": r.get("instagram", ""),
            "ig_post": r.get("ig_post", ""), "llm_reason": r.get("llm_reason", ""),
            "review_snippet": r.get("review_snippet", ""),
        })
    mpts = pd.DataFrame(rows)
    mpts = mpts[(mpts.lat.between(51.2, 51.8)) & (mpts.lon.between(-0.6, 0.4))] \
        if not mpts.empty else mpts
    if not mpts.empty:
        st.caption("📍 Results on the map — hover for a name, click a dot for the card.")
        picked = _venue_map(mpts, key="search_map")
        if picked:
            st.markdown("**📍 Selected on map**")
            with st.container(border=True):
                _render_card(picked)
    # ── Bucket 1: matches everything you asked for ──────────────────────
    if alternatives:
        st.markdown(f"### ✅ Matches what you asked for  ·  {len(matches)}")
        st.caption("These satisfy every requirement in your search.")
    else:
        st.markdown(f"**{len(matches)} places** for *{qy}*")
    st.divider()
    if matches:
        for r in matches:
            _render_card(r)
            st.divider()
    else:
        st.info("Nothing matched every requirement exactly — see the suggestions below.")

    # ── Bucket 2: good fits that miss / can't-confirm one requirement ────
    if alternatives:
        st.markdown(f"### 💡 Also worth a look  ·  {len(alternatives)}")
        st.caption("Strong matches that miss — or that we can't confirm satisfy — one of your requirements.")
        st.divider()
        for r in alternatives:
            _render_card(r, is_alt=True)

    # ── Bucket 3: lazy-load tail — browse more options on demand ─────────
    if more:
        with st.expander(f"➕ Show {len(more)} more places that fit", expanded=False):
            st.caption("Ranked by match but not individually re-scored — broaden your choice.")
            for r in more:
                _render_card(r)
                st.divider()
            st.divider()

# ── Feedback: report a miss / suggest a fix → feeds the learning loop ────────
st.divider()
with st.expander("💬 Spotted a miss? Help improve Gustave", expanded=False):
    st.caption("If a search missed a place it should have shown, tell us — it goes "
               "straight into Gustave's learning loop.")
    with st.form("feedback_form", clear_on_submit=True):
        fb_q = st.text_input("The search you ran", value=st.session_state.get("_query", ""))
        fb_v = st.text_input("Place(s) that should have appeared (comma-separated)")
        fb_n = st.text_area("Anything else? (optional)", height=70)
        sent = st.form_submit_button("Send feedback")
    if sent:
        venues = [v.strip() for v in fb_v.split(",") if v.strip()]
        if not (fb_q.strip() and venues):
            st.warning("Add the search and at least one place that should have appeared.")
        else:
            try:
                import learn_core, cloud_log
                from datetime import datetime
                entry = {
                    "id": datetime.now().strftime("%Y-%m-%d-%H%M%S"),
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "query": fb_q.strip(), "should_surface": venues,
                    "note": fb_n.strip(), "source": "live", "status": "pending", "triage": None}
                learn_core.append_learning(entry)          # local (ephemeral on cloud)
                cloud_log.append("learnings.jsonl", entry)  # durable mirror to gist
                st.success("Thanks — logged! Andrei will take a look.")
            except Exception as e:
                st.caption(f"Couldn't save right now: {e}")

st.caption("Gustave is an early test build — results and links are still being "
           "refined. Spotted something off? Tell Andrei.")
