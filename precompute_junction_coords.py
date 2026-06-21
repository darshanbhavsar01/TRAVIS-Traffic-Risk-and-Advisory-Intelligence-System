"""
Integration 3 — Junction Geocoding (one-time pre-computation)

Runs every junction name from the Astram lookup tables through the Mappls
Geocode API and stores the results in models/junction_coords.json. The
Streamlit app then places each prediction's map pin at the exact junction
coordinate instead of the corridor centroid.

Usage:
    python precompute_junction_coords.py            # geocode missing only
    python precompute_junction_coords.py --refresh  # re-geocode everything

Credentials are read from the .env file (CLIENT_ID / CLIENT_SECRET).
"""

import os
import sys
import json
import time

import requests
from dotenv import load_dotenv

load_dotenv()

MODEL_DIR    = "models"
LOOKUP_PATH  = f"{MODEL_DIR}/lookup_tables.json"
OUT_PATH     = f"{MODEL_DIR}/junction_coords.json"

CLIENT_ID     = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")

# Rough Bengaluru bounding box — used to discard bad geocodes.
BLR_BBOX = {"lat_min": 12.7, "lat_max": 13.25, "lon_min": 77.3, "lon_max": 77.9}


def get_token():
    resp = requests.post(
        "https://outpost.mappls.com/api/security/oauth/token",
        data={
            "grant_type": "client_credentials",
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def clean_name(raw):
    """Turn 'AgaraJunction' / '28thMainJayanagarJunc' into a friendlier query."""
    import re
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", raw)          # camelCase -> spaced
    s = s.replace("Jn", " Junction").replace("Junc", " Junction")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def geocode(token, name):
    query = f"{clean_name(name)}, Bengaluru, Karnataka"
    try:
        resp = requests.get(
            "https://atlas.mappls.com/api/places/geocode",
            params={"address": query, "region": "IND", "itemCount": 1},
            headers={"Authorization": f"Bearer {token}"},
            timeout=15,
        )
        if resp.status_code != 200:
            return None
        body = resp.json()
        results = body.get("copResults") or body.get("results")
        if isinstance(results, dict):
            results = [results]
        if not results:
            return None
        r = results[0]
        lat = r.get("latitude") or r.get("lat")
        lon = r.get("longitude") or r.get("lng") or r.get("lon")
        if lat is None or lon is None:
            return None
        lat, lon = float(lat), float(lon)
        if not (BLR_BBOX["lat_min"] <= lat <= BLR_BBOX["lat_max"]
                and BLR_BBOX["lon_min"] <= lon <= BLR_BBOX["lon_max"]):
            return None  # geocoded outside Bengaluru — reject
        return {"lat": round(lat, 6), "lon": round(lon, 6),
                "query": query, "formatted": r.get("formattedAddress", "")}
    except Exception as e:
        print(f"   ! error: {e}")
        return None


def main():
    if not CLIENT_ID or not CLIENT_SECRET:
        sys.exit("CLIENT_ID / CLIENT_SECRET missing — set them in .env")

    refresh = "--refresh" in sys.argv

    with open(LOOKUP_PATH, encoding="utf-8") as f:
        lookup = json.load(f)
    junctions = sorted(lookup["junction"].keys())
    print(f"Found {len(junctions)} junctions in lookup tables.")

    existing = {}
    if os.path.exists(OUT_PATH) and not refresh:
        with open(OUT_PATH, encoding="utf-8") as f:
            existing = json.load(f)
        print(f"Loaded {len(existing)} previously geocoded junctions.")

    todo = [j for j in junctions if j not in existing]
    print(f"Geocoding {len(todo)} junctions "
          f"({'refresh' if refresh else 'missing only'})...\n")

    token = get_token()
    ok, fail = len(existing), 0

    for i, name in enumerate(todo, 1):
        coords = geocode(token, name)
        if coords:
            existing[name] = coords
            ok += 1
            print(f"[{i}/{len(todo)}] ✓ {name} -> {coords['lat']},{coords['lon']}")
        else:
            fail += 1
            print(f"[{i}/{len(todo)}] ✗ {name} (no result)")

        # Persist every 25 calls so progress survives interruption.
        if i % 25 == 0:
            with open(OUT_PATH, "w", encoding="utf-8") as f:
                json.dump(existing, f, indent=2, ensure_ascii=False)
        time.sleep(0.15)  # be gentle with the API

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)

    print(f"\nDone. {ok} geocoded, {fail} failed this run.")
    print(f"Written to {OUT_PATH}")


if __name__ == "__main__":
    main()
