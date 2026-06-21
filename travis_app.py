"""
TRAVIS - Traffic Risk & Advisory Intelligence System
Bangalore Event-Driven Traffic Congestion Prediction
Phase 4: Streamlit Demo App
"""

import os
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
import math
import requests
import streamlit.components.v1 as components
from datetime import datetime, date, time
import warnings
warnings.filterwarnings("ignore")

# Load environment variables from .env file
load_dotenv()

# ─────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TRAVIS — Bangalore Traffic Intelligence",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────
# GLOBAL STYLE  (dark theme matching notebooks)
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Main background */
  .stApp { background-color: #0f1117; }
  section[data-testid="stSidebar"] { background-color: #1a1d27; border-right: 1px solid #2a2d3a; }

  /* Cards */
  .travis-card {
    background: #1a1d27; border: 1px solid #2a2d3a; border-radius: 12px;
    padding: 20px 24px; margin-bottom: 16px;
  }
  .metric-card {
    background: #1a1d27; border: 1px solid #2a2d3a; border-radius: 10px;
    padding: 18px 20px; text-align: center;
  }
  .metric-label { font-size: 12px; color: #8b8fa8; font-weight: 500; letter-spacing: 0.04em; margin-bottom: 6px; }
  .metric-value { font-size: 30px; font-weight: 700; }
  .metric-sub   { font-size: 11px; color: #8b8fa8; margin-top: 4px; }

  /* Score gauge */
  .score-ring {
    display: flex; flex-direction: column; align-items: center;
    justify-content: center; padding: 24px;
  }
  .score-number { font-size: 80px; font-weight: 800; line-height: 1; }
  .score-label  { font-size: 14px; color: #8b8fa8; letter-spacing: 0.08em; margin-top: 8px; }
  .score-tier   { font-size: 22px; font-weight: 700; margin-top: 4px; }

  /* Tiers */
  .tier-HIGH   { color: #ff6b6b; }
  .tier-MEDIUM { color: #ffd166; }
  .tier-LOW    { color: #00d4aa; }

  /* Deployment card table */
  .deploy-row { display: flex; gap: 12px; margin-bottom: 10px; flex-wrap: wrap; }
  .deploy-chip {
    background: #0f1117; border: 1px solid #2a2d3a; border-radius: 8px;
    padding: 10px 16px; flex: 1; min-width: 100px; text-align: center;
  }
  .deploy-chip-label { font-size: 10px; color: #8b8fa8; letter-spacing: 0.06em; }
  .deploy-chip-value { font-size: 22px; font-weight: 700; color: #e8eaf0; }
  .deploy-chip-unit  { font-size: 10px; color: #8b8fa8; }

  /* SHAP bars */
  .shap-bar-pos { background: linear-gradient(90deg, #ff6b6b88, #ff6b6b); border-radius: 3px; height: 18px; }
  .shap-bar-neg { background: linear-gradient(90deg, #6c63ff88, #6c63ff); border-radius: 3px; height: 18px; }

  /* Scenario buttons */
  div[data-testid="column"] .stButton > button {
    width: 100%; border-radius: 8px; font-size: 12px; padding: 8px 6px;
    border: 1px solid #2a2d3a; background: #1a1d27; color: #e8eaf0;
    transition: all 0.2s;
  }
  div[data-testid="column"] .stButton > button:hover {
    border-color: #6c63ff; background: #1f2237; color: #fff;
  }

  /* Header */
  .travis-header {
    display: flex; align-items: center; gap: 16px;
    padding: 20px 0 16px; border-bottom: 1px solid #2a2d3a; margin-bottom: 24px;
  }
  .travis-title { font-size: 28px; font-weight: 800; color: #e8eaf0; }
  .travis-sub   { font-size: 13px; color: #8b8fa8; margin-top: 2px; }

  /* Section titles */
  .section-title {
    font-size: 11px; font-weight: 700; color: #8b8fa8;
    letter-spacing: 0.12em; text-transform: uppercase;
    margin-bottom: 12px; margin-top: 4px;
  }

  /* Streamlit overrides */
  .stSelectbox label, .stSlider label, .stDateInput label,
  .stTimeInput label, .stRadio label { color: #8b8fa8 !important; font-size: 12px !important; }
  div[data-testid="stMetricValue"] { color: #e8eaf0; }
  .stSpinner > div { border-top-color: #6c63ff !important; }
  hr { border-color: #2a2d3a !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────
# LOAD MODELS & LOOKUPS  (cached)
# ─────────────────────────────────────────────────────────────────────
MODEL_DIR = "models"

@st.cache_resource
def load_models():
    clf_rc  = joblib.load(f"{MODEL_DIR}/model_road_closure.pkl")
    clf_pri = joblib.load(f"{MODEL_DIR}/model_priority.pkl")
    reg_res = joblib.load(f"{MODEL_DIR}/model_resolution.pkl")
    meta    = json.load(open(f"{MODEL_DIR}/model_meta.json"))
    lookup  = json.load(open(f"{MODEL_DIR}/lookup_tables.json"))
    return clf_rc, clf_pri, reg_res, meta, lookup

clf_rc, clf_pri, reg_res, MODEL_META, LOOKUP = load_models()
FEATURES   = MODEL_META["features"]
THR_RC     = MODEL_META["threshold_rc"]
RES_P95    = MODEL_META["res_p95"]
RULES      = MODEL_META["resource_rules"]
CAUSE_RANK = LOOKUP["cause_impact_rank"]
T0         = pd.Timestamp(LOOKUP["t0"])

# ─────────────────────────────────────────────────────────────────────
# MAPPLS (Map My India) API INTEGRATION
# ─────────────────────────────────────────────────────────────────────
MAPPLS_CLIENT_ID     = os.getenv("CLIENT_ID")
MAPPLS_CLIENT_SECRET = os.getenv("CLIENT_SECRET")

# Known coordinates for major Bangalore police stations
PS_COORDS = {
    "Sadashivanagar":  {"lat": 13.0085, "lon": 77.5832},
    "Vijayanagara":    {"lat": 12.9693, "lon": 77.5372},
    "Hebbala":         {"lat": 13.0375, "lon": 77.6182},
    "Peenya":          {"lat": 13.0305, "lon": 77.5178},
    "Upparpet":        {"lat": 12.9657, "lon": 77.5741},
    "Malleshwaram":    {"lat": 13.0020, "lon": 77.5683},
    "Rajajinagar":     {"lat": 12.9962, "lon": 77.5525},
    "Banashankari":    {"lat": 12.9261, "lon": 77.5656},
    "Yeshwanthpura":   {"lat": 13.0246, "lon": 77.5406},
    "Whitefield":      {"lat": 12.9698, "lon": 77.7500},
    "Koramangala":     {"lat": 12.9352, "lon": 77.6245},
    "Indiranagar":     {"lat": 12.9784, "lon": 77.6408},
    "HSR Layout":      {"lat": 12.9116, "lon": 77.6474},
    "Byatarayanapura": {"lat": 13.0657, "lon": 77.5688},
    "Jnanabharathi":   {"lat": 12.9490, "lon": 77.5086},
    "Kengeri":         {"lat": 12.9112, "lon": 77.4827},
    "Yelahanka":       {"lat": 13.1007, "lon": 77.5963},
    "Electronic City": {"lat": 12.8430, "lon": 77.6750},
    "Mahadevapura":    {"lat": 12.9937, "lon": 77.6882},
    "KR Puram":        {"lat": 13.0035, "lon": 77.6960},
    "Marathahalli":    {"lat": 12.9591, "lon": 77.7007},
    "Bellandur":       {"lat": 12.9260, "lon": 77.6762},
    "Hebbal":          {"lat": 13.0353, "lon": 77.5970},
    "Govindpura":      {"lat": 13.0224, "lon": 77.6046},
    "Basavanagudi":    {"lat": 12.9420, "lon": 77.5684},
    "Jayanagar":       {"lat": 12.9302, "lon": 77.5838},
    "JP Nagar":        {"lat": 12.9082, "lon": 77.5836},
    "BTM Layout":      {"lat": 12.9166, "lon": 77.6101},
    "Hulimavu":        {"lat": 12.8873, "lon": 77.6147},
    "Bommanahalli":    {"lat": 12.9070, "lon": 77.6306},
}


@st.cache_data(ttl=3300)
def get_mappls_token():
    """Fetch Mappls OAuth token (cached 55 min)."""
    try:
        resp = requests.post(
            "https://outpost.mappls.com/api/security/oauth/token",
            data={
                "grant_type": "client_credentials",
                "client_id": MAPPLS_CLIENT_ID,
                "client_secret": MAPPLS_CLIENT_SECRET,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=10,
        )
        if resp.status_code == 200:
            return resp.json().get("access_token", "")
    except Exception:
        pass
    return ""


@st.cache_data(ttl=600)
def get_route_data(token, origin_lat, origin_lon, dest_lat, dest_lon):
    """Fetch primary + alternate routes from Mappls Route API."""
    if not token:
        return None
    try:
        resp = requests.get(
            "https://apis.mappls.com/advancedmaps/v1/route_adv/json",
            params={
                "origin": f"{origin_lat},{origin_lon}",
                "destination": f"{dest_lat},{dest_lon}",
                "alternatives": "true",
                "region": "IND",
            },
            headers={"Authorization": f"Bearer {token}"},
            timeout=10,
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


@st.cache_data(ttl=600)
def get_nearest_stations(token, event_lat, event_lon):
    """Find 3 nearest police stations by road distance (Distance Matrix API)."""
    def straight_dist(ps):
        dlat = PS_COORDS[ps]["lat"] - event_lat
        dlon = PS_COORDS[ps]["lon"] - event_lon
        return dlat ** 2 + dlon ** 2

    candidates = sorted(PS_COORDS.keys(), key=straight_dist)[:6]

    if token:
        try:
            dest_str = "|".join(
                f"{PS_COORDS[ps]['lat']},{PS_COORDS[ps]['lon']}" for ps in candidates
            )
            resp = requests.get(
                "https://apis.mappls.com/advancedmaps/v1/distance_matrix/json",
                params={
                    "origins": f"{event_lat},{event_lon}",
                    "destinations": dest_str,
                    "region": "IND",
                },
                headers={"Authorization": f"Bearer {token}"},
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                rows = (data.get("results", {}).get("rows")
                        or data.get("rows", []))
                elems = rows[0].get("elements", []) if rows else []
                results = []
                for ps, elem in zip(candidates, elems):
                    dist_val = (elem.get("distance") or {}).get("value", 0)
                    if dist_val > 0:
                        results.append({"name": ps, "distance_km": round(dist_val / 1000, 1)})
                if results:
                    return sorted(results, key=lambda x: x["distance_km"])[:3]
        except Exception:
            pass

    # Fallback: straight-line (≈111 km/degree)
    return sorted(
        [{"name": ps, "distance_km": round(straight_dist(ps) ** 0.5 * 111, 1)} for ps in candidates],
        key=lambda x: x["distance_km"],
    )[:3]


# ── Junction geocoding (Integration 3) ──────────────────────────────
@st.cache_resource
def load_junction_coords():
    """Load pre-computed junction → coordinate lookup (built by
    precompute_junction_coords.py). Returns {} if not yet generated."""
    try:
        with open(f"{MODEL_DIR}/junction_coords.json", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


JUNCTION_COORDS = load_junction_coords()


@st.cache_data(ttl=86400)
def geocode_junction(token, name):
    """Resolve a single junction name to precise coords via the Mappls
    Geocode API. Cached 24h. Used as a live fallback when a junction is
    not in the pre-computed junction_coords.json."""
    if not token or not name:
        return None
    try:
        resp = requests.get(
            "https://atlas.mappls.com/api/places/geocode",
            params={"address": f"{name}, Bengaluru, Karnataka", "region": "IND",
                    "itemCount": 1},
            headers={"Authorization": f"Bearer {token}"},
            timeout=10,
        )
        if resp.status_code == 200:
            results = resp.json().get("copResults") or resp.json().get("results")
            if isinstance(results, dict):
                results = [results]
            if results:
                r = results[0]
                lat = r.get("latitude") or r.get("lat")
                lon = r.get("longitude") or r.get("lng") or r.get("lon")
                if lat and lon:
                    return {"lat": float(lat), "lon": float(lon)}
    except Exception:
        pass
    return None


def resolve_junction_coords(token, junction, fallback_lat, fallback_lon):
    """Return precise junction coords: pre-computed → live geocode → corridor centroid."""
    if not junction:
        return fallback_lat, fallback_lon, "corridor centroid"
    jc = JUNCTION_COORDS.get(junction)
    if jc and jc.get("lat") and jc.get("lon"):
        return jc["lat"], jc["lon"], "geocoded (precomputed)"
    live = geocode_junction(token, junction)
    if live:
        return live["lat"], live["lon"], "geocoded (live)"
    return fallback_lat, fallback_lon, "corridor centroid"


def _decode_polyline(encoded):
    path, i, lat, lng = [], 0, 0, 0
    while i < len(encoded):
        b, shift, result = 0, 0, 0
        while True:
            b = ord(encoded[i]) - 63
            i += 1
            result |= (b & 0x1f) << shift
            shift += 5
            if b < 0x20:
                break
        lat += ~(result >> 1) if (result & 1) else (result >> 1)
        shift, result = 0, 0
        while True:
            b = ord(encoded[i]) - 63
            i += 1
            result |= (b & 0x1f) << shift
            shift += 5
            if b < 0x20:
                break
        lng += ~(result >> 1) if (result & 1) else (result >> 1)
        path.append({"lat": round(lat / 1e5, 6), "lng": round(lng / 1e5, 6)})
    return path


def build_route_coords(route_data, clat, clon):
    """Extract route coordinate arrays; fall back to synthetic polylines."""
    def synthetic():
        off, det = 0.025, 0.018
        return {
            "primary": [
                {"lat": round(clat - off, 6), "lng": round(clon, 6)},
                {"lat": round(clat - off * 0.4, 6), "lng": round(clon + 0.003, 6)},
                {"lat": round(clat, 6), "lng": round(clon, 6)},
                {"lat": round(clat + off * 0.4, 6), "lng": round(clon - 0.003, 6)},
                {"lat": round(clat + off, 6), "lng": round(clon, 6)},
            ],
            "alternate": [
                {"lat": round(clat - off, 6), "lng": round(clon, 6)},
                {"lat": round(clat - off * 0.5, 6), "lng": round(clon - det, 6)},
                {"lat": round(clat, 6), "lng": round(clon - det, 6)},
                {"lat": round(clat + off * 0.5, 6), "lng": round(clon - det, 6)},
                {"lat": round(clat + off, 6), "lng": round(clon, 6)},
            ],
            "alternate2": [],
        }

    if route_data is None:
        return synthetic()

    def extract(geom):
        if not geom:
            return []
        if isinstance(geom, str) and len(geom) > 4:
            try:
                return _decode_polyline(geom)
            except Exception:
                return []
        if isinstance(geom, dict) and "coordinates" in geom:
            return [{"lat": c[1], "lng": c[0]} for c in geom["coordinates"]]
        if isinstance(geom, list) and geom and isinstance(geom[0], dict):
            return geom
        return []

    # Handle both {results:{trips:[...]}} and {routes:[...]} formats
    trips = route_data.get("results", {}).get("trips", [])
    if trips:
        raw = [t.get("polyline", "") for t in trips]
    else:
        raw = [r.get("geometry", "") for r in route_data.get("routes", [])]

    pts = [extract(g) for g in raw]
    pts = [p for p in pts if len(p) >= 2]

    if not pts:
        return synthetic()

    return {
        "primary":   pts[0],
        "alternate":  pts[1] if len(pts) > 1 else synthetic()["alternate"],
        "alternate2": pts[2] if len(pts) > 2 else [],
    }


def extract_route_stats(route_data, route_coords):
    """Return a stats dict for the ETA comparison card.

    Uses real Mappls API duration/distance when available; falls back to
    computing distance from polyline geometry at a typical urban speed.
    """
    URBAN_KMH = 25  # km/h — conservative Bangalore urban speed

    def _haversine(lat1, lon1, lat2, lon2):
        p = math.pi / 180
        a = (0.5 - math.cos((lat2 - lat1) * p) / 2
             + math.cos(lat1 * p) * math.cos(lat2 * p)
             * (1 - math.cos((lon2 - lon1) * p)) / 2)
        return 2 * 6371 * math.asin(math.sqrt(a))

    def _path_km(pts):
        if not pts or len(pts) < 2:
            return 0.0
        return round(
            sum(_haversine(pts[i-1]["lat"], pts[i-1]["lng"],
                           pts[i]["lat"],   pts[i]["lng"])
                for i in range(1, len(pts))), 2)

    source = "estimated"
    p_dur_s = p_dist_m = a_dur_s = a_dist_m = 0

    if route_data:
        # Format 1 — {results: {trips: [{duration, length, polyline}, ...]}}
        trips = route_data.get("results", {}).get("trips", [])
        if trips and trips[0].get("duration", 0) > 0:
            p = trips[0]
            a = trips[1] if len(trips) > 1 else {}
            p_dur_s, p_dist_m = p.get("duration", 0), p.get("length", 0)
            a_dur_s, a_dist_m = a.get("duration", 0), a.get("length", 0)
            source = "api"
        else:
            # Format 2 — {routes: [{duration, distance, geometry}, ...]}
            routes = route_data.get("routes", [])
            if routes and routes[0].get("duration", 0) > 0:
                p = routes[0]
                a = routes[1] if len(routes) > 1 else {}
                p_dur_s, p_dist_m = p.get("duration", 0), p.get("distance", 0)
                a_dur_s, a_dist_m = a.get("duration", 0), a.get("distance", 0)
                source = "api"

    if source == "estimated":
        p_km     = _path_km(route_coords.get("primary", []))
        a_km     = _path_km(route_coords.get("alternate", []))
        p_dist_m = p_km * 1000
        a_dist_m = a_km * 1000
        p_dur_s  = (p_km / URBAN_KMH) * 3600
        a_dur_s  = (a_km / URBAN_KMH) * 3600

    p_min = max(1, round(p_dur_s / 60))
    a_min = max(1, round(a_dur_s / 60))
    p_km  = round(p_dist_m / 1000, 1)
    a_km  = round(a_dist_m / 1000, 1)

    return {
        "primary_min": p_min,
        "primary_km":  p_km,
        "divert_min":  a_min,
        "divert_km":   a_km,
        "delta_min":   max(0, a_min - p_min),
        "delta_km":    round(max(0.0, a_km - p_km), 1),
        "source":      source,
    }


def build_mappls_map_html(token, clat, clon, tier, tc, radius, impact,
                          sel_corridor, sel_cause, route_coords, ps_markers):
    """Return self-contained HTML embedding the Mappls JS SDK map."""
    fill_opacity = 0.25 if tier == "HIGH" else 0.18 if tier == "MEDIUM" else 0.10
    route_json   = json.dumps(route_coords)
    ps_json      = json.dumps(ps_markers)
    corr_js      = json.dumps(sel_corridor)
    cause_js     = json.dumps(sel_cause.replace("_", " ").title())

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
html, body {{ width:100%; height:100%; background:#0f1117; }}
#map {{ width:100%; height:420px; border-radius:12px; overflow:hidden; }}
#map-error {{
    display:none; width:100%; height:420px; border-radius:12px;
    background:#1a1d27; border:1px solid #2a2d3a;
    align-items:center; justify-content:center; flex-direction:column; gap:8px;
    color:#8b8fa8; font-family:sans-serif; font-size:14px; text-align:center;
}}
.map-legend {{
    position:absolute; bottom:32px; left:12px; z-index:10;
    background:rgba(15,17,23,0.88); border:1px solid #2a2d3a; border-radius:8px;
    padding:8px 12px; font-family:sans-serif; font-size:11px; color:#e8eaf0;
    pointer-events:none;
}}
.leg {{ display:flex; align-items:center; gap:6px; margin-bottom:3px; }}
.leg:last-child {{ margin-bottom:0; }}
.dot {{ width:10px; height:10px; border-radius:50%; flex-shrink:0; }}
.bar {{ width:18px; height:4px; border-radius:2px; flex-shrink:0; }}
</style>
</head>
<body>
<div id="map"></div>
<div id="map-error">
  <div style="font-size:28px;">⚠️</div>
  <div>Mappls map unavailable</div>
  <div style="font-size:11px;color:#555;margin-top:4px;">Check API credentials or network connection</div>
</div>
<script>
var TOKEN        = '{token}';
var LAT          = {clat};
var LON          = {clon};
var TIER         = '{tier}';
var COLOR        = '{tc}';
var RAD          = {radius};
var SCORE        = {impact};
var FILL_OPACITY = {fill_opacity};
var CORR         = {corr_js};
var CAUSE        = {cause_js};
var ROUTES       = {route_json};
var PS_MARKERS   = {ps_json};

function showError() {{
    document.getElementById('map').style.display = 'none';
    document.getElementById('map-error').style.display = 'flex';
}}

function decodePolyline(enc) {{
    var path = [], i = 0, lat = 0, lng = 0;
    while (i < enc.length) {{
        var b, shift = 0, res = 0;
        do {{ b = enc.charCodeAt(i++) - 63; res |= (b & 0x1f) << shift; shift += 5; }} while (b >= 0x20);
        lat += (res & 1) ? ~(res >> 1) : (res >> 1);
        shift = 0; res = 0;
        do {{ b = enc.charCodeAt(i++) - 63; res |= (b & 0x1f) << shift; shift += 5; }} while (b >= 0x20);
        lng += (res & 1) ? ~(res >> 1) : (res >> 1);
        path.push({{lat: lat / 1e5, lng: lng / 1e5}});
    }}
    return path;
}}

function getPoints(geom) {{
    if (!geom || !geom.length) return [];
    if (typeof geom === 'string') return decodePolyline(geom);
    if (geom[0] && geom[0].lat !== undefined) return geom;
    if (geom[0] && Array.isArray(geom[0])) return geom.map(function(c) {{ return {{lat:c[1], lng:c[0]}}; }});
    return [];
}}

function drawLine(map, pts, color, weight, opacity) {{
    if (!pts || pts.length < 2) return;
    new mappls.Polyline({{map:map, path:pts, strokeColor:color, strokeWeight:weight, strokeOpacity:opacity||0.85}});
}}

function addLegend(hasRoutes) {{
    var d = document.createElement('div');
    d.className = 'map-legend';
    var h = '<div class="leg"><div class="dot" style="background:' + COLOR + ';opacity:0.75;"></div><span>Impact zone (' + RAD + 'm)</span></div>';
    h    += '<div class="leg"><div class="dot" style="background:#3498db;"></div><span>Police station</span></div>';
    if (hasRoutes) {{
        h += '<div class="leg"><div class="bar" style="background:#ff4444;"></div><span>Primary route (blocked)</span></div>';
        h += '<div class="leg"><div class="bar" style="background:#00d4aa;"></div><span>Diversion route</span></div>';
    }}
    h += '<div class="leg" style="margin-top:4px;border-top:1px solid #2a2d3a22;padding-top:4px;"><span style="color:#555;font-size:10px;">Mappls SDK · Live traffic · Lane-level data</span></div>';
    d.innerHTML = h;
    document.getElementById('map').appendChild(d);
}}

function initMap() {{
    try {{
        var map = new mappls.Map('map', {{center:{{lat:LAT, lng:LON}}, zoom:13}});
        map.on('load', function() {{
            var hasRoutes = false;

            try {{ mappls.trafficLayer({{map:map, show:true}}); }} catch(e) {{}}

            try {{
                new mappls.Circle({{
                    map:map, center:new mappls.LatLng(LAT, LON), radius:RAD,
                    strokeColor:COLOR, strokeOpacity:0.9, strokeWeight:2,
                    fillColor:COLOR, fillOpacity:FILL_OPACITY
                }});
            }} catch(e) {{}}

            try {{
                var p  = getPoints(ROUTES.primary);
                var a1 = getPoints(ROUTES.alternate);
                var a2 = getPoints(ROUTES.alternate2);
                if (p.length >= 2)  {{ drawLine(map, p,  '#ff4444', 5, 0.75); hasRoutes = true; }}
                if (a1.length >= 2) {{ drawLine(map, a1, '#00d4aa', 4, 0.85); }}
                if (a2.length >= 2) {{ drawLine(map, a2, '#ffd166', 3, 0.75); }}
            }} catch(e) {{}}

            try {{
                new mappls.Marker({{
                    map:map, position:{{lat:LAT, lng:LON}}, draggable:false,
                    popupHtml:'<div style="font-family:sans-serif;min-width:160px;padding:4px;">' +
                              '<b style="color:' + COLOR + ';">' + CAUSE + '</b><br>' +
                              '<span style="color:#888;">Corridor: </span>' + CORR + '<br>' +
                              '<span style="color:#888;">Impact: </span><b>' + SCORE + '/100 · ' + TIER + '</b></div>'
                }});
            }} catch(e) {{}}

            try {{
                (PS_MARKERS || []).forEach(function(ps) {{
                    new mappls.Marker({{
                        map:map, position:{{lat:ps.lat, lng:ps.lon}},
                        popupHtml:'<b>&#x1F6E1;&#xFE0F; ' + ps.name + ' PS</b>' +
                                  (ps.dist ? '<br><span style="color:#888;">' + ps.dist + ' km by road</span>' : '')
                    }});
                }});
            }} catch(e) {{}}

            addLegend(hasRoutes);
        }});
    }} catch(err) {{ showError(); }}
}}

(function() {{
    if (!TOKEN) {{ showError(); return; }}
    var s = document.createElement('script');
    s.src = 'https://apis.mappls.com/advancedmaps/api/' + TOKEN + '/map_sdk?layer=vector&v=3.0&callback=initMap';
    s.onerror = showError;
    document.head.appendChild(s);
}})();
</script>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING  (mirrors Phase 2 exactly)
# ─────────────────────────────────────────────────────────────────────
def engineer_features(event_cause, corridor, police_station, event_dt,
                      event_type="unplanned", junction=None):
    """Build the 32-feature vector from user inputs using lookup tables."""
    ist_dt   = pd.Timestamp(event_dt, tz="Asia/Kolkata")
    hour     = ist_dt.hour
    dow_ord  = ist_dt.dayofweek          # Mon=0 … Sun=6
    month_n  = ist_dt.month
    days_ss  = max(0, (ist_dt - T0.tz_convert("Asia/Kolkata")).days)

    # Cause lookups
    c_lu     = LOOKUP["cause"].get(event_cause, {})
    cause_clos  = c_lu.get("cause_closure_rate_hist", LOOKUP["defaults"].get("cause_closure_rate_hist", 0.08))
    cause_rank  = CAUSE_RANK.get(event_cause, 0)

    # Corridor lookups
    corr_lu  = LOOKUP["corridor"].get(corridor, {})
    gm       = LOOKUP["global_medians"]
    corr_clos  = corr_lu.get("corr_closure_rt",         LOOKUP["defaults"].get("corr_closure_rt", 0.08))
    corr_hpri  = corr_lu.get("corr_high_prio_rt",       LOOKUP["defaults"].get("corr_high_prio_rt", 0.6))
    corr_frag  = corr_lu.get("corridor_fragility",      LOOKUP["defaults"].get("corridor_fragility", 20))
    corr_fn    = corr_lu.get("corridor_fragility_norm",  LOOKUP["defaults"].get("corridor_fragility_norm", 50))
    corr_te_r  = corr_lu.get("corridor_te_requir",       0.08)
    corr_te_p  = corr_lu.get("corridor_te_priori",       0.60)

    # Police station lookups
    ps_lu    = LOOKUP["police_station"].get(police_station, {})
    ps_hs    = ps_lu.get("ps_hotspot_score",        LOOKUP["defaults"].get("ps_hotspot_score", 0.35))
    ps_clos  = ps_lu.get("ps_closure_rt",            LOOKUP["defaults"].get("ps_closure_rt", 0.08))
    ps_hpri  = ps_lu.get("ps_highprio_rt",           LOOKUP["defaults"].get("ps_highprio_rt", 0.60))
    ps_mres  = ps_lu.get("ps_med_res",                LOOKUP["defaults"].get("ps_med_res", 1.0))
    ps_te_r  = ps_lu.get("police_station_te_requir",  0.08)
    ps_te_p  = ps_lu.get("police_station_te_priori",  0.60)

    # Zone lookup (via corridor)
    zone_name = LOOKUP["corr_zone"].get(corridor)
    z_lu      = LOOKUP["zone"].get(zone_name, {}) if zone_name else {}
    z_clos    = z_lu.get("zone_closure_rt",  gm["zone_closure_rt"])
    z_risk    = z_lu.get("zone_risk_score",  gm["zone_risk_score"])

    # Junction lookup — use exact junction history when one is selected,
    # otherwise fall back to global medians.
    j_lu     = LOOKUP["junction"].get(junction, {}) if junction else {}
    j_clos   = j_lu.get("junc_closure_rt",        gm["junc_closure_rt"])
    j_hpri   = j_lu.get("junc_highprio_rt",       gm["junc_highprio_rt"])
    j_hs     = j_lu.get("junction_hotspot_score", gm["junction_hotspot_score"])
    j_total  = j_lu.get("junc_total",             gm["junc_total"])

    # Vehicle — event-driven causes have no vehicle
    veh_clos = 0.0
    veh_risk = 0.0

    # Interactions
    inter_cc  = round(corr_clos * cause_clos, 4)
    inter_pc  = round(ps_hs * cause_rank / 5.0, 4)
    corr_fn01 = round(corr_fn / 100.0, 4)

    feat = {
        "cause_closure_rate_hist":         cause_clos,
        "cause_pot_holes":                 int(event_cause == "pot_holes"),
        "corr_closure_rt":                 corr_clos,
        "corr_high_prio_rt":               corr_hpri,
        "corridor_fragility":              corr_frag,
        "corridor_fragility_norm":         corr_fn,
        "corridor_te_priori":              corr_te_p,
        "corridor_te_requir":              corr_te_r,
        "days_since_start":                days_ss,
        "dow_cos":                         round(np.cos(2*np.pi*dow_ord/7), 6),
        "dow_ord":                         dow_ord,
        "dow_sin":                         round(np.sin(2*np.pi*dow_ord/7), 6),
        "event_type_enc":                  int(event_type == "planned"),
        "hour_cos":                        round(np.cos(2*np.pi*hour/24), 6),
        "hour_sin":                        round(np.sin(2*np.pi*hour/24), 6),
        "interaction_corr_cause_closure":  inter_cc,
        "interaction_ps_cause":            inter_pc,
        "junc_closure_rt":                 j_clos,
        "junc_highprio_rt":                j_hpri,
        "junc_total":                      j_total,
        "junction_hotspot_score":          j_hs,
        "month_num":                       month_n,
        "police_station_te_priori":        ps_te_p,
        "police_station_te_requir":        ps_te_r,
        "ps_closure_rt":                   ps_clos,
        "ps_highprio_rt":                  ps_hpri,
        "ps_hotspot_score":                ps_hs,
        "ps_med_res":                      ps_mres,
        "veh_closure_rt":                  veh_clos,
        "veh_risk_score":                  veh_risk,
        "zone_closure_rt":                 z_clos,
        "zone_risk_score":                 z_risk,
    }
    return pd.DataFrame([{k: feat[k] for k in FEATURES}])


def predict(X, event_cause_key="others"):
    p_rc  = clf_rc.predict_proba(X)[:,1][0]
    p_pri = clf_pri.predict_proba(X)[:,1][0]
    pred_log_hrs = reg_res.predict(X)[0]
    pred_hrs = float(np.clip(np.expm1(pred_log_hrs), 0, None))
    res_norm  = min(pred_hrs / RES_P95, 1.0)
    cr_norm   = CAUSE_RANK.get(event_cause_key, 0) / 5.0
    impact    = round(p_rc*55 + cr_norm*30 + res_norm*15, 1)
    tier      = "HIGH" if impact >= 50 else "MEDIUM" if impact >= 25 else "LOW"
    rule     = RULES[tier]
    return {
        "p_closure":   round(p_rc  * 100, 1),
        "p_priority":  round(p_pri * 100, 1),
        "pred_hrs":    round(pred_hrs, 1),
        "impact":      impact,
        "tier":        tier,
        "constables":  rule["constables"],
        "barricades":  rule["barricades"],
        "response_min":rule["response_time_min"],
    }


def get_shap_values(X):
    try:
        import shap
        explainer = shap.TreeExplainer(clf_rc)
        sv = explainer.shap_values(X)
        sv = sv[1] if isinstance(sv, list) else sv
        return sv[0], explainer.expected_value[1] if hasattr(explainer.expected_value,'__len__') else explainer.expected_value
    except Exception:
        return None, None


# ─────────────────────────────────────────────────────────────────────
# PREDEFINED SCENARIOS
# ─────────────────────────────────────────────────────────────────────
SCENARIOS = {
    "🔴 VIP Movement – Bellary Rd": {
        "event_cause": "vip_movement",
        "corridor": "Bellary Road 1",
        "police_station": "Sadashivanagar",
        "hour": 9,
        "dow": "Tuesday",
        "event_type": "planned",
        "label": "VIP motorcade, Tuesday 9 AM",
    },
    "🟠 Public Rally – Mysore Rd": {
        "event_cause": "public_event",
        "corridor": "Mysore Road",
        "police_station": "Vijayanagara",
        "hour": 17,
        "dow": "Friday",
        "event_type": "planned",
        "label": "Public rally, Friday 5 PM",
    },
    "🟡 Construction – ORR North": {
        "event_cause": "construction",
        "corridor": "ORR North 1",
        "police_station": "Hebbala",
        "hour": 8,
        "dow": "Monday",
        "event_type": "unplanned",
        "label": "Road work, Monday 8 AM",
    },
    "⚪ Breakdown – Tumkur Rd": {
        "event_cause": "vehicle_breakdown",
        "corridor": "Tumkur Road",
        "police_station": "Peenya",
        "hour": 14,
        "dow": "Wednesday",
        "event_type": "unplanned",
        "label": "Truck breakdown, Wednesday 2 PM",
    },
    "🟠 Procession – CBD": {
        "event_cause": "procession",
        "corridor": "CBD 2",
        "police_station": "Upparpet",
        "hour": 10,
        "dow": "Sunday",
        "event_type": "planned",
        "label": "Religious procession, Sunday 10 AM",
    },
}

DOW_OPTIONS = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
DOW_MAP     = {d: i for i, d in enumerate(DOW_OPTIONS)}
CAUSE_OPTIONS = [
    "vip_movement","public_event","procession","protest",
    "construction","congestion","accident","vehicle_breakdown",
    "tree_fall","water_logging","pot_holes","others",
]
CAUSE_EMOJI = {
    "vip_movement":"🚨","public_event":"🎪","procession":"🎭","protest":"✊",
    "construction":"🚧","congestion":"🚗","accident":"💥","vehicle_breakdown":"🚛",
    "tree_fall":"🌳","water_logging":"🌊","pot_holes":"🕳️","others":"❓",
}

# ─────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────
if "scenario" not in st.session_state:
    st.session_state.scenario = None
if "result"   not in st.session_state:
    st.session_state.result   = None
if "feat_vec" not in st.session_state:
    st.session_state.feat_vec = None
if "junction" not in st.session_state:
    st.session_state.junction = None

# ─────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="travis-header">
  <span style="font-size:42px;">🚦</span>
  <div>
    <div class="travis-title">TRAVIS</div>
    <div class="travis-sub">Traffic Risk & Advisory Intelligence System &nbsp;·&nbsp; Bangalore</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────
# SIDEBAR — INPUT FORM
# ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📋 Event Parameters")
    st.markdown("---")

    # Quick scenarios
    st.markdown('<div class="section-title">Quick Scenarios</div>', unsafe_allow_html=True)
    scenario_cols = st.columns(1)
    for sc_name, sc_data in SCENARIOS.items():
        if st.button(sc_name, key=f"sc_{sc_name}", width="stretch"):
            st.session_state.scenario = sc_data

    st.markdown("---")
    st.markdown('<div class="section-title">Custom Event</div>', unsafe_allow_html=True)

    # If a scenario was selected, pre-fill the form
    sc = st.session_state.scenario or {}

    sel_cause = st.selectbox(
        "Event Type",
        options=CAUSE_OPTIONS,
        index=CAUSE_OPTIONS.index(sc.get("event_cause","public_event")),
        format_func=lambda x: f"{CAUSE_EMOJI.get(x,'⚪')}  {x.replace('_',' ').title()}",
    )
    sel_corridor = st.selectbox(
        "Corridor",
        options=sorted(LOOKUP["corridor"].keys()),
        index=sorted(LOOKUP["corridor"].keys()).index(sc.get("corridor","Mysore Road")),
    )
    sel_ps = st.selectbox(
        "Police Station",
        options=sorted(LOOKUP["police_station"].keys()),
        index=sorted(LOOKUP["police_station"].keys()).index(sc.get("police_station","Vijayanagara")),
    )
    JUNCTION_NONE = "— Auto (corridor centre) —"
    junction_opts = [JUNCTION_NONE] + sorted(LOOKUP["junction"].keys())
    sel_junction_raw = st.selectbox(
        "Junction (optional — precise pin)",
        options=junction_opts,
        index=junction_opts.index(sc.get("junction", JUNCTION_NONE))
              if sc.get("junction", JUNCTION_NONE) in junction_opts else 0,
        help="Pick a named junction to geocode an exact pin via Mappls and "
             "use that junction's incident history in the prediction.",
    )
    sel_junction = None if sel_junction_raw == JUNCTION_NONE else sel_junction_raw
    sel_dow = st.selectbox(
        "Day of Week",
        options=DOW_OPTIONS,
        index=DOW_OPTIONS.index(sc.get("dow","Friday")),
    )
    sel_hour = st.slider(
        "Hour of Day (IST)",
        min_value=0, max_value=23,
        value=sc.get("hour", 17),
        format="%d:00",
    )
    sel_type = st.radio(
        "Event Classification",
        options=["planned","unplanned"],
        index=0 if sc.get("event_type","planned")=="planned" else 1,
        horizontal=True,
    )

    st.markdown("---")
    predict_btn = st.button("🔮  Forecast Impact", width="stretch", type="primary")

# ─────────────────────────────────────────────────────────────────────
# RUN PREDICTION
# ─────────────────────────────────────────────────────────────────────
if predict_btn:
    dow_n   = DOW_MAP[sel_dow]
    base_dt = datetime(2024, 3, 4+dow_n, sel_hour, 0, 0)  # reference week
    X = engineer_features(
        event_cause    = sel_cause,
        corridor       = sel_corridor,
        police_station = sel_ps,
        event_dt       = base_dt,
        event_type     = sel_type,
        junction       = sel_junction,
    )
    st.session_state.result   = predict(X, event_cause_key=sel_cause)
    st.session_state.feat_vec = X
    st.session_state.junction = sel_junction

# ─────────────────────────────────────────────────────────────────────
# RESULTS LAYOUT
# ─────────────────────────────────────────────────────────────────────
res = st.session_state.result

if res is None:
    # Landing state
    st.markdown("""
    <div class="travis-card" style="text-align:center; padding: 60px 40px;">
      <div style="font-size:64px; margin-bottom:16px;">🗺️</div>
      <div style="font-size:20px; font-weight:700; color:#e8eaf0; margin-bottom:10px;">
        Select an event and click Forecast Impact
      </div>
      <div style="font-size:14px; color:#8b8fa8; max-width:500px; margin:0 auto; line-height:1.6;">
        TRAVIS predicts road closure probability, priority level, and disruption duration
        for any event on Bangalore's traffic network — then generates a resource
        deployment plan in real time.
      </div>
      <div style="margin-top:32px; display:flex; gap:16px; justify-content:center; flex-wrap:wrap;">
        <div style="background:#1a1d27;border:1px solid #2a2d3a;border-radius:10px;padding:16px 24px;">
          <div style="font-size:28px;font-weight:800;color:#00d4aa;">3</div>
          <div style="font-size:11px;color:#8b8fa8;">ML Models</div>
        </div>
        <div style="background:#1a1d27;border:1px solid #2a2d3a;border-radius:10px;padding:16px 24px;">
          <div style="font-size:28px;font-weight:800;color:#6c63ff;">8,171</div>
          <div style="font-size:11px;color:#8b8fa8;">Training Events</div>
        </div>
        <div style="background:#1a1d27;border:1px solid #2a2d3a;border-radius:10px;padding:16px 24px;">
          <div style="font-size:28px;font-weight:800;color:#ffd166;">22</div>
          <div style="font-size:11px;color:#8b8fa8;">Corridors Mapped</div>
        </div>
        <div style="background:#1a1d27;border:1px solid #2a2d3a;border-radius:10px;padding:16px 24px;">
          <div style="font-size:28px;font-weight:800;color:#ff6b6b;">54</div>
          <div style="font-size:11px;color:#8b8fa8;">Police Stations</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

else:
    tier   = res["tier"]
    impact = res["impact"]

    TIER_COLOR = {"HIGH":"#ff6b6b","MEDIUM":"#ffd166","LOW":"#00d4aa"}
    TIER_LABEL = {"HIGH":"🔴 HIGH RISK","MEDIUM":"🟡 MEDIUM RISK","LOW":"🟢 LOW RISK"}
    tc = TIER_COLOR[tier]

    # ── ROW 1: Score gauge + 4 metrics ───────────────────────────────
    col_score, col_metrics = st.columns([1, 2.5])

    with col_score:
        st.markdown(f"""
        <div class="travis-card" style="text-align:center; padding:32px 20px; border-color:{tc}44;">
          <div class="section-title" style="text-align:center;">IMPACT SCORE</div>
          <div class="score-number" style="color:{tc};">{impact:.0f}</div>
          <div style="color:#8b8fa8; font-size:13px; margin:4px 0 12px;">out of 100</div>

          <div style="background:#0f1117; border-radius:999px; height:8px; margin:0 12px 16px;">
            <div style="background:{tc}; width:{min(impact,100)}%; height:8px; border-radius:999px;
                        box-shadow: 0 0 12px {tc}88;"></div>
          </div>

          <div class="score-tier" style="color:{tc};">{TIER_LABEL[tier]}</div>

          <div style="margin-top:20px; display:flex; justify-content:space-around;
                      padding:12px 0; border-top:1px solid #2a2d3a; border-bottom:1px solid #2a2d3a;">
            <div style="text-align:center;">
              <div style="font-size:10px;color:#8b8fa8;">CLOSURE</div>
              <div style="font-size:16px;font-weight:700;color:#ff6b6b;">40%</div>
            </div>
            <div style="text-align:center;">
              <div style="font-size:10px;color:#8b8fa8;">PRIORITY</div>
              <div style="font-size:16px;font-weight:700;color:#ffd166;">30%</div>
            </div>
            <div style="text-align:center;">
              <div style="font-size:10px;color:#8b8fa8;">DURATION</div>
              <div style="font-size:16px;font-weight:700;color:#6c63ff;">30%</div>
            </div>
          </div>
          <div style="font-size:10px;color:#8b8fa8;margin-top:10px;">impact score weights</div>
        </div>
        """, unsafe_allow_html=True)

    with col_metrics:
        m1, m2, m3, m4 = st.columns(4)

        p_clos_color = "#ff6b6b" if res["p_closure"] > 50 else "#ffd166" if res["p_closure"] > 25 else "#00d4aa"
        p_pri_color  = "#ff6b6b" if res["p_priority"] > 70 else "#ffd166"

        for mcol, label, value, unit, color, sub in [
            (m1, "P(Road Closure)", f"{res['p_closure']:.1f}",  "%",  p_clos_color, "Probability closure required"),
            (m2, "P(High Priority)",f"{res['p_priority']:.1f}", "%",  p_pri_color,  "Probability high priority"),
            (m3, "Est. Duration",    f"{res['pred_hrs']:.1f}",  "hrs",  "#a8edea",  "Predicted resolution time"),
            (m4, "Event Type",      sel_type.upper(),           "",   "#8b8fa8",    f"{CAUSE_EMOJI.get(sel_cause,'⚪')} {sel_cause.replace('_',' ').title()}"),
        ]:
            mcol.markdown(f"""
            <div class="metric-card">
              <div class="metric-label">{label}</div>
              <div class="metric-value" style="color:{color};">{value}<span style="font-size:16px;font-weight:400;color:#8b8fa8;"> {unit}</span></div>
              <div class="metric-sub">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

        # Progress bars for closure & priority
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        pb1, pb2 = st.columns(2)
        with pb1:
            st.markdown(f"<div style='font-size:11px;color:#8b8fa8;margin-bottom:4px;'>Road Closure Probability</div>", unsafe_allow_html=True)
            st.progress(int(res["p_closure"]))
        with pb2:
            st.markdown(f"<div style='font-size:11px;color:#8b8fa8;margin-bottom:4px;'>Priority High Probability</div>", unsafe_allow_html=True)
            st.progress(int(res["p_priority"]))

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── ROW 2: Deployment Plan + Map ─────────────────────────────────
    coords     = LOOKUP["corr_coords"].get(sel_corridor, {"lat": 12.9716, "lon": 77.5946})
    corr_lat, corr_lon = coords["lat"], coords["lon"]
    RADIUS     = {"HIGH": 2000, "MEDIUM": 1200, "LOW": 600}
    MAP_COLOR  = {"HIGH": "#ff6b6b", "MEDIUM": "#ffd166", "LOW": "#00d4aa"}

    # Mappls API calls (cached — fast on repeat runs)
    token      = get_mappls_token()

    # Integration 3 — resolve precise junction pin (geocoded) when a junction
    # was selected, else fall back to the corridor centroid.
    cur_junction = st.session_state.get("junction")
    clat, clon, pin_source = resolve_junction_coords(token, cur_junction, corr_lat, corr_lon)

    nearest_ps = get_nearest_stations(token, clat, clon)
    ps_markers = [
        {"name": ps["name"], "lat": PS_COORDS[ps["name"]]["lat"],
         "lon": PS_COORDS[ps["name"]]["lon"], "dist": ps["distance_km"]}
        for ps in nearest_ps if ps["name"] in PS_COORDS
    ]
    # Route: nearest police station → incident location.
    # Semantics: "time for nearest unit to reach the blocked junction" —
    # changes meaningfully per corridor and gives the diversion real context.
    if ps_markers:
        ps0 = ps_markers[0]
        r_orig_lat, r_orig_lon = ps0["lat"], ps0["lon"]
    else:
        r_orig_lat, r_orig_lon = round(clat - 0.027, 6), clon
    route_data   = get_route_data(token, r_orig_lat, r_orig_lon, clat, clon)
    route_coords = build_route_coords(route_data, clat, clon)
    route_stats  = extract_route_stats(route_data, route_coords)

    col_deploy, col_map = st.columns([1, 1.6])

    with col_deploy:
        st.markdown(f"""
        <div class="travis-card" style="border-color:{tc}44;">
          <div class="section-title">Resource Deployment Plan</div>
          <div class="deploy-row">
            <div class="deploy-chip">
              <div class="deploy-chip-label">CONSTABLES</div>
              <div class="deploy-chip-value" style="color:{tc};">{res['constables']}</div>
              <div class="deploy-chip-unit">officers</div>
            </div>
            <div class="deploy-chip">
              <div class="deploy-chip-label">BARRICADES</div>
              <div class="deploy-chip-value" style="color:{tc};">{res['barricades']}</div>
              <div class="deploy-chip-unit">points</div>
            </div>
            <div class="deploy-chip">
              <div class="deploy-chip-label">RESPONSE</div>
              <div class="deploy-chip-value" style="color:{tc};">{res['response_min']}</div>
              <div class="deploy-chip-unit">min</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Nearest deployment sources (Mappls Distance Matrix)
        if nearest_ps:
            rows_html = ""
            for i, ps in enumerate(nearest_ps, 1):
                rows_html += (
                    f'<div style="display:flex;justify-content:space-between;align-items:center;'
                    f'padding:6px 0;border-bottom:1px solid #2a2d3a22;">'
                    f'<span style="font-size:12px;color:#e8eaf0;">({i}) {ps["name"]} PS</span>'
                    f'<span style="font-size:12px;font-weight:600;color:#00d4aa;">{ps["distance_km"]} km</span>'
                    f'</div>'
                )
            st.markdown(f"""
            <div class="travis-card" style="border-color:{tc}22;margin-top:0;">
              <div class="section-title">Nearest Deployment Sources</div>
              {rows_html}
              <div style="font-size:10px;color:#555;margin-top:6px;">
                Road distance · Mappls Distance Matrix API
              </div>
            </div>
            """, unsafe_allow_html=True)

        # Corridor fragility info
        corr_lu = LOOKUP["corridor"].get(sel_corridor, {})
        cf_norm = corr_lu.get("corridor_fragility_norm", 50)
        cf_color = "#ff6b6b" if cf_norm > 65 else "#ffd166" if cf_norm > 35 else "#00d4aa"

        st.markdown(f"""
        <div class="travis-card">
          <div class="section-title">Corridor Intelligence</div>
          <div style="margin-bottom:12px;">
            <div style="font-size:12px;color:#8b8fa8;margin-bottom:4px;">Fragility Index</div>
            <div style="background:#0f1117;border-radius:999px;height:6px;">
              <div style="background:{cf_color};width:{cf_norm}%;height:6px;border-radius:999px;"></div>
            </div>
            <div style="display:flex;justify-content:space-between;margin-top:4px;">
              <span style="font-size:11px;color:#8b8fa8;">Low</span>
              <span style="font-size:12px;font-weight:700;color:{cf_color};">{cf_norm:.0f}/100</span>
              <span style="font-size:11px;color:#8b8fa8;">High</span>
            </div>
          </div>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;font-size:12px;">
            <div style="color:#8b8fa8;">Historical Closure Rate</div>
            <div style="color:#e8eaf0;text-align:right;font-weight:600;">{corr_lu.get('corr_closure_rt',0)*100:.1f}%</div>
            <div style="color:#8b8fa8;">High Priority Rate</div>
            <div style="color:#e8eaf0;text-align:right;font-weight:600;">{corr_lu.get('corr_high_prio_rt',0)*100:.1f}%</div>
            <div style="color:#8b8fa8;">Avg Resolution</div>
            <div style="color:#e8eaf0;text-align:right;font-weight:600;">{corr_lu.get('corr_med_res',1):.1f} hrs</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Action checklist
        actions = {
            "HIGH":   ["☎️ Alert duty officer immediately",
                       "🚧 Deploy barricades at junction",
                       "📢 Activate BMTC rerouting",
                       "📡 Broadcast advisory on VMS boards",
                       "🚨 Request additional units from HQ"],
            "MEDIUM": ["📋 Notify police station in-charge",
                       "🚧 Pre-position barricades",
                       "📍 Activate spot monitoring",
                       "📢 Issue precautionary advisory"],
            "LOW":    ["📋 Log event for monitoring",
                       "👁️ Assign routine patrol",
                       "📊 Track for pattern analysis"],
        }
        st.markdown(f"""
        <div class="travis-card">
          <div class="section-title">Action Checklist</div>
          {''.join(f'<div style="font-size:13px;color:#e8eaf0;padding:5px 0;border-bottom:1px solid #2a2d3a22;">{a}</div>' for a in actions[tier])}
        </div>
        """, unsafe_allow_html=True)

    with col_map:
        st.markdown(
            '<div class="section-title">📍 Impact Map — Bangalore &nbsp;·&nbsp; '
            '<span style="color:#6c63ff;">Powered by Mappls</span></div>',
            unsafe_allow_html=True,
        )

        map_html = build_mappls_map_html(
            token=token, clat=clat, clon=clon,
            tier=tier, tc=tc,
            radius=RADIUS[tier],
            impact=impact,
            sel_corridor=sel_corridor,
            sel_cause=sel_cause,
            route_coords=route_coords,
            ps_markers=ps_markers,
        )
        components.html(map_html, height=440, scrolling=False)

        # ── ETA / Diversion comparison card (Integration 2 — Route API) ──
        rs = route_stats
        if rs["divert_min"] > 0:
            # Operational verdict
            if rs["delta_min"] <= 5:
                verdict_color, verdict_icon, verdict_text = (
                    "#00d4aa", "✅", "Minimal overhead — viable rerouting")
            elif rs["delta_min"] <= 15:
                verdict_color, verdict_icon, verdict_text = (
                    "#ffd166", "⚠️", "Moderate detour — advise early diversion")
            else:
                verdict_color, verdict_icon, verdict_text = (
                    "#ff6b6b", "🚨", "Major detour — coordinate BMTC & VMS boards")

            ps_origin_name = ps_markers[0]["name"] if ps_markers else "nearest PS"
            src_note = (
                f'<span style="color:#555;font-size:10px;">'
                f'{ps_origin_name} → incident · Mappls API</span>'
                if rs["source"] == "api" else
                f'<span style="color:#555;font-size:10px;">'
                f'{ps_origin_name} → incident · estimated</span>'
            )
            delta_str = (
                f'+{rs["delta_min"]} min · +{rs["delta_km"]} km'
                if rs["delta_km"] > 0
                else f'+{rs["delta_min"]} min'
            )

            _eta_html = (
                f'<div class="travis-card" style="border-color:{tc}33;margin-top:0;padding:16px 20px;">'
                f'<div class="section-title">🚔 Police Response — ETA to Incident</div>'
                f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:12px;">'
                f'<div style="background:#0f1117;border:1px solid #ff444440;border-radius:8px;padding:11px 14px;">'
                f'<div style="font-size:9px;color:#ff4444;letter-spacing:0.07em;margin-bottom:3px;">DIRECT ROUTE (BLOCKED)</div>'
                f'<div style="display:flex;align-items:baseline;gap:4px;">'
                f'<span style="font-size:26px;font-weight:800;color:#ff4444;line-height:1;">{rs["primary_min"]}</span>'
                f'<span style="font-size:12px;color:#8b8fa8;"> min</span>'
                f'</div>'
                f'<div style="font-size:11px;color:#8b8fa8;margin-top:2px;">{rs["primary_km"]} km</div>'
                f'</div>'
                f'<div style="background:#0f1117;border:1px solid #00d4aa40;border-radius:8px;padding:11px 14px;">'
                f'<div style="font-size:9px;color:#00d4aa;letter-spacing:0.07em;margin-bottom:3px;">VIA DIVERSION</div>'
                f'<div style="display:flex;align-items:baseline;gap:4px;">'
                f'<span style="font-size:26px;font-weight:800;color:#00d4aa;line-height:1;">{rs["divert_min"]}</span>'
                f'<span style="font-size:12px;color:#8b8fa8;"> min</span>'
                f'</div>'
                f'<div style="font-size:11px;color:#8b8fa8;margin-top:2px;">{rs["divert_km"]} km</div>'
                f'</div>'
                f'</div>'
                f'<div style="background:#0f1117;border:1px solid {verdict_color}33;border-radius:8px;padding:10px 14px;display:flex;align-items:center;gap:10px;">'
                f'<div style="font-size:20px;line-height:1;">{verdict_icon}</div>'
                f'<div style="flex:1;">'
                f'<div style="font-size:13px;font-weight:700;color:{verdict_color};">Detour: {delta_str}</div>'
                f'<div style="font-size:11px;color:#8b8fa8;margin-top:2px;">{verdict_text}</div>'
                f'</div>'
                f'<div style="text-align:right;">{src_note}</div>'
                f'</div>'
                f'</div>'
            )
            st.markdown(_eta_html, unsafe_allow_html=True)

        # Pin provenance (Integration 3)
        if cur_junction and "geocoded" in pin_source:
            st.markdown(
                f'<div style="font-size:11px;color:#00d4aa;margin-top:2px;">'
                f'📍 Pin at exact junction: <b>{cur_junction}</b> '
                f'<span style="color:#555;">({clat:.4f}, {clon:.4f} · {pin_source})</span></div>',
                unsafe_allow_html=True,
            )
        elif cur_junction:
            st.markdown(
                f'<div style="font-size:11px;color:#ffd166;margin-top:2px;">'
                f'📍 "{cur_junction}" could not be geocoded — pin at corridor centre</div>',
                unsafe_allow_html=True,
            )

        if token:
            st.markdown(
                '<div style="font-size:10px;color:#555;text-align:right;margin-top:2px;">'
                '🗺️ Mappls SDK &nbsp;·&nbsp; 🚦 Live traffic layer'
                ' &nbsp;·&nbsp; 🛣️ Route API &nbsp;·&nbsp; 📍 Lane-level Bangalore data'
                '</div>',
                unsafe_allow_html=True,
            )

    # ── ROW 3: SHAP Explanation ────────────────────────────────────────
    st.markdown("---")
    col_shap, col_hist = st.columns([1.2, 1])

    with col_shap:
        st.markdown('<div class="section-title">🧠 Why this prediction? (SHAP Explanation)</div>', unsafe_allow_html=True)

        X = st.session_state.feat_vec
        sv, base = get_shap_values(X)

        if sv is not None:
            feat_df = pd.DataFrame({
                "feature": FEATURES,
                "shap":    sv,
                "value":   X.values[0],
            }).reindex(pd.Series(np.abs(sv)).sort_values(ascending=False).index)
            feat_df = feat_df.head(10)

            max_shap = max(abs(feat_df["shap"].max()), abs(feat_df["shap"].min()), 0.01)

            st.markdown('<div class="travis-card">', unsafe_allow_html=True)
            for _, row in feat_df.iterrows():
                s      = row["shap"]
                width  = abs(s) / max_shap * 100
                color  = "#ff6b6b" if s > 0 else "#6c63ff"
                arrow  = "▲" if s > 0 else "▼"
                impact_label = "increases closure risk" if s > 0 else "reduces closure risk"
                feat_name = row["feature"].replace("_"," ")

                st.markdown(f"""
                <div style="margin-bottom:12px;">
                  <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                    <span style="font-size:12px;color:#e8eaf0;">{feat_name[:30]}</span>
                    <span style="font-size:11px;color:{color};font-weight:600;">
                      {arrow} {abs(s):.3f} — {impact_label}
                    </span>
                  </div>
                  <div style="background:#0f1117;border-radius:4px;height:10px;">
                    <div style="background:{color};width:{width:.1f}%;height:10px;
                                border-radius:4px;box-shadow:0 0 6px {color}66;"></div>
                  </div>
                  <div style="font-size:10px;color:#555;margin-top:2px;">
                    value = {row['value']:.4f}
                  </div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            st.caption(f"Base rate (all events): {base:.1%} → "
                       f"This event: {res['p_closure']/100:.1%} road closure probability")
        else:
            st.info("SHAP explanation unavailable — install shap library to enable.")

    with col_hist:
        st.markdown('<div class="section-title">📊 Historical Baseline — This Cause</div>', unsafe_allow_html=True)

        c_lu_hist = LOOKUP["cause"].get(sel_cause, {})
        all_causes_clos = {k: v.get("cause_closure_rate_hist", 0)
                          for k, v in LOOKUP["cause"].items()
                          if v.get("count", 0) > 10}
        df_plot = pd.DataFrame(all_causes_clos.items(), columns=["cause","closure_rate"])
        df_plot = df_plot.sort_values("closure_rate", ascending=True)

        CAUSE_COLORS_PLOT = {
            "vip_movement":"#ff6b6b","public_event":"#ff6b6b","procession":"#ff6b6b",
            "protest":"#ff6b6b","construction":"#ffd166","congestion":"#ffd166",
        }

        chart_data = df_plot.set_index("cause")["closure_rate"].mul(100)
        bar_colors = [
            CAUSE_COLORS_PLOT.get(c, "#6c63ff" if c == sel_cause else "#2a2d3a")
            for c in chart_data.index
        ]

        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")

        fig, ax = plt.subplots(figsize=(6, 4.5))
        fig.patch.set_facecolor("#1a1d27")
        ax.set_facecolor("#1a1d27")

        bars = ax.barh(chart_data.index, chart_data.values,
                       color=bar_colors, edgecolor="none", height=0.65)
        for bar, val in zip(bars, chart_data.values):
            ax.text(bar.get_width()+0.3, bar.get_y()+bar.get_height()/2,
                    f"{val:.0f}%", va="center", fontsize=8, color="#8b8fa8")

        # Highlight selected cause
        sel_idx = list(chart_data.index).index(sel_cause) if sel_cause in chart_data.index else -1
        if sel_idx >= 0:
            bars[sel_idx].set_color(tc)
            bars[sel_idx].set_linewidth(1.5)

        ax.set_xlabel("Historical Road Closure Rate (%)", color="#8b8fa8", fontsize=9)
        ax.set_title(f"Closure Rate by Cause\n(selected: {sel_cause.replace('_',' ')})",
                     color="#e8eaf0", fontsize=9, pad=8)
        ax.tick_params(colors="#8b8fa8", labelsize=8)
        ax.spines[:].set_color("#2a2d3a")
        ax.grid(axis="x", alpha=0.2, color="#2a2d3a")

        plt.tight_layout()
        st.pyplot(fig, width="stretch")
        plt.close()

        # Key stats
        st.markdown(f"""
        <div class="travis-card" style="margin-top:12px;">
          <div class="section-title">Event Context</div>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;font-size:12px;">
            <div style="color:#8b8fa8;">Cause</div>
            <div style="color:#e8eaf0;font-weight:600;">{CAUSE_EMOJI.get(sel_cause,'')} {sel_cause.replace('_',' ').title()}</div>
            <div style="color:#8b8fa8;">Corridor</div>
            <div style="color:#e8eaf0;font-weight:600;">{sel_corridor}</div>
            <div style="color:#8b8fa8;">Day / Time</div>
            <div style="color:#e8eaf0;font-weight:600;">{sel_dow}  {sel_hour:02d}:00</div>
            <div style="color:#8b8fa8;">Classification</div>
            <div style="color:#e8eaf0;font-weight:600;">{sel_type.upper()}</div>
            <div style="color:#8b8fa8;">Impact Rank</div>
            <div style="color:{tc};font-weight:700;">{CAUSE_RANK.get(sel_cause,0)}/5</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#555;font-size:11px;padding:12px 0;">
  TRAVIS &nbsp;·&nbsp; Hackathon Prototype &nbsp;·&nbsp;
  Models trained on Astram incident data Nov 2023–Apr 2024 &nbsp;·&nbsp;
  Recommendations are advisory — operator override applies
</div>
""", unsafe_allow_html=True)
