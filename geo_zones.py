"""
Gustave — London geo zones (location-aware search support)
==========================================================
Pure, import-light helpers for the OPTIONAL location restrict step. No model /
no API — it post-filters an already-ranked result set, so it's cheap enough to
run on a cached base result. Disabled by default (see search_v2.LOCATION_ENABLED).

Design: the expensive search is location-agnostic. This module then:
  1. extract_location(query)  — heuristically reads the requested area/zone from
     the raw query (no LLM), so a cached base can be reused across locations.
  2. strip_location(query)    — removes the location phrase for the cache key, so
     "italian in Soho" and "italian in Shoreditch" share one cached base.
  3. partition_by_location()  — tags each result in_location / near_location and
     reorders: in the requested area first, then same-zone nearby, then the rest.
Zone boundaries are deliberately coarse and easy to tune.
"""
from __future__ import annotations

import math

ZONES = ("central", "west", "east", "north", "south")

# Neighbourhood → broad zone. Lowercase keys; matched as substrings of an address
# or the raw query. Extend freely — this is the tuning surface.
AREA_TO_ZONE = {
    # central
    "soho": "central", "covent garden": "central", "mayfair": "central",
    "fitzrovia": "central", "marylebone": "central", "bloomsbury": "central",
    "holborn": "central", "westminster": "central", "st james": "central",
    "victoria": "central", "pimlico": "central", "clerkenwell": "central",
    "farringdon": "central", "bankside": "central", "borough": "central",
    "waterloo": "central", "south bank": "central", "southbank": "central",
    "the city": "central", "aldgate": "central", "moorgate": "central",
    "king's cross": "central", "kings cross": "central", "euston": "central",
    # west
    "notting hill": "west", "kensington": "west", "chelsea": "west",
    "hammersmith": "west", "fulham": "west", "shepherd's bush": "west",
    "shepherds bush": "west", "bayswater": "west", "paddington": "west",
    "ealing": "west", "chiswick": "west", "maida vale": "west",
    "earls court": "west", "earl's court": "west", "white city": "west",
    "acton": "west", "hampstead": "west",  # NW-ish; treat as west for now
    # east
    "shoreditch": "east", "hackney": "east", "dalston": "east",
    "bethnal green": "east", "hoxton": "east", "spitalfields": "east",
    "whitechapel": "east", "canary wharf": "east", "bow": "east",
    "stratford": "east", "hackney wick": "east", "london fields": "east",
    "clapton": "east", "leyton": "east", "bermondsey": "east",
    "wapping": "east", "shadwell": "east", "homerton": "east",
    # north
    "islington": "north", "camden": "north", "angel": "north",
    "highbury": "north", "stoke newington": "north", "crouch end": "north",
    "highgate": "north", "kentish town": "north", "archway": "north",
    "finsbury park": "north", "tufnell park": "north", "holloway": "north",
    "muswell hill": "north", "newington green": "north",
    # south
    "brixton": "south", "peckham": "south", "clapham": "south",
    "battersea": "south", "vauxhall": "south", "greenwich": "south",
    "deptford": "south", "new cross": "south", "dulwich": "south",
    "camberwell": "south", "elephant and castle": "south", "stockwell": "south",
    "tooting": "south", "balham": "south", "wandsworth": "south",
    "putney": "south", "nine elms": "south", "kennington": "south",
}

# Phrases that name a broad zone directly.
_ZONE_PHRASES = {
    "central london": "central", "the city": "central", "city of london": "central",
    "west london": "west", "west end": "central",
    "east london": "east",
    "north london": "north",
    "south london": "south", "south east london": "south", "south west london": "south",
}

_PREPOSITIONS = ("in", "near", "around", "by", "close to", "next to")

# Coarse coordinate box for central London (≈ within ~2.5km of Charing Cross).
_C_LAT_LO, _C_LAT_HI = 51.490, 51.535
_C_LON_LO, _C_LON_HI = -0.160, -0.070


def _num(x):
    try:
        f = float(x)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


# ── 1. read the requested location from the raw query ───────────────────────────

def extract_location(query: str) -> dict | None:
    """Heuristically read the requested area/zone from the raw query (no LLM).
    Returns {"raw","zone","area"} or None. `area` is a specific neighbourhood
    (or None for a bare zone request)."""
    q = " " + (query or "").lower() + " "
    # Zone phrases first (longest match wins).
    for phrase in sorted(_ZONE_PHRASES, key=len, reverse=True):
        if f" {phrase} " in q or q.strip().endswith(phrase):
            return {"raw": phrase, "zone": _ZONE_PHRASES[phrase], "area": None}
    # Then specific neighbourhoods.
    for area in sorted(AREA_TO_ZONE, key=len, reverse=True):
        if area in q:
            return {"raw": area, "zone": AREA_TO_ZONE[area], "area": area}
    return None


def strip_location(query: str) -> str:
    """Remove the matched location phrase (and a trailing preposition) from the
    query, for a location-agnostic cache key."""
    loc = extract_location(query)
    if not loc:
        return query
    out = " " + (query or "") + " "
    low = out.lower()
    phrase = loc["raw"]
    idx = low.find(phrase)
    if idx == -1:
        return query
    before, after = out[:idx], out[idx + len(phrase):]
    # drop a dangling preposition just before the location ("... in soho")
    b = before.rstrip()
    for prep in sorted(_PREPOSITIONS, key=len, reverse=True):
        if b.lower().endswith(" " + prep):
            b = b[: -(len(prep) + 1)]
            break
    return " ".join((b + " " + after).split()).strip()


# ── 2. classify a venue into a zone ─────────────────────────────────────────────

def venue_zone(lat, lon, address: str = "") -> str | None:
    """Best-effort zone for a venue: prefer a named area in its address, else
    fall back to coarse coordinate boxes. Returns a zone name or None."""
    addr = (address or "").lower()
    for area, z in AREA_TO_ZONE.items():
        if area in addr:
            return z
    la, lo = _num(lat), _num(lon)
    if la is None or lo is None:
        return None
    if _C_LAT_LO <= la <= _C_LAT_HI and _C_LON_LO <= lo <= _C_LON_HI:
        return "central"
    if lo < _C_LON_LO:
        return "west"
    if lo > _C_LON_HI:
        return "east"
    if la > _C_LAT_HI:
        return "north"
    if la < _C_LAT_LO:
        return "south"
    return "central"


def _in_area(row: dict, area: str) -> bool:
    return area in str(row.get("address", "")).lower()


# ── 3. partition an already-ranked result set by location ───────────────────────

def partition_by_location(results: list, loc: dict) -> list:
    """Tag each result and reorder: requested area first, then same-zone nearby,
    then everything else (preserving the base score order within each band).
    Returns NEW dicts (does not mutate the cached base)."""
    zone = loc.get("zone")
    area = loc.get("area")
    in_loc, near, far = [], [], []
    for r in results:
        vz = venue_zone(r.get("latitude"), r.get("longitude"), r.get("address", ""))
        if area:
            is_in = _in_area(r, area)
            is_near = (not is_in) and (vz == zone)
        else:  # bare zone request
            is_in = (vz == zone)
            is_near = False
        tagged = {**r, "in_location": bool(is_in), "near_location": bool(is_near), "venue_zone": vz}
        (in_loc if is_in else near if is_near else far).append(tagged)
    return in_loc + near + far
