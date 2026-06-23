# TRAVIS — Traffic Risk & Advisory Intelligence System

Event-driven traffic congestion prediction and advisory for Bangalore. TRAVIS forecasts
road-closure risk, priority, and disruption duration for any event on the city's traffic
network, then produces a deployment plan, an interactive Mappls map with live diversion
routing, an AI-written advisory, and a one-click public bulletin.

> **Architecture & flow diagrams:** see **[DIAGRAMS.md](DIAGRAMS.md)**.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Create a .env file in this folder with your API keys
#    CLIENT_ID=<Mappls client id>
#    CLIENT_SECRET=<Mappls client secret>
#    GROQ_API_KEY=<Groq api key>

# 3. (Optional, one-time) precompute exact junction coordinates
python precompute_junction_coords.py

# 4. Run the app
streamlit run travis_app.py
```

Expected folder structure:

```
phase4_app/
  models/
    model_road_closure.pkl
    model_priority.pkl
    model_resolution.pkl
    model_meta.json
    lookup_tables.json
    junction_coords.json   # optional, built by precompute_junction_coords.py
  travis_app.py
  precompute_junction_coords.py
  requirements.txt
  .env                     # API keys (gitignored)
```

## What the App Does

| Feature | Detail |
|---------|--------|
| Impact Score | 0–100 composite from 3 ML models (weights 55 / 30 / 15) |
| Road Closure | LightGBM binary classifier (AUC 0.72) |
| Priority | LightGBM binary classifier (AUC 0.9995) |
| Duration | LightGBM regressor, log-transformed target |
| SHAP | Per-prediction feature attribution (top risk drivers) |
| Map | Mappls JS SDK — lane-level data + live traffic layer + impact radius |
| Diversion Routing | Mappls Route API — blocked vs diversion polyline + ETA |
| Nearest Units | Mappls Distance Matrix API — nearest police stations by road |
| Junction Pins | Mappls Geocode API — exact named-junction coordinates |
| Deployment Plan | Rule engine: officers, barricades, response time |
| AI Advisory | Groq `llama-3.3-70b` — 3-sentence advisory from model data only |
| Public Bulletin | One-click printable HTML/PDF advisory for commuters |

## External Integrations

- **Mappls (Map My India):** OAuth token, JS SDK + traffic layer, Route API,
  Distance Matrix API, Geocode API, Static Map API. Every call degrades gracefully
  (synthetic route / straight-line distance / corridor centroid) if the API is
  unavailable.
- **Groq LLM:** receives only model-derived facts (prediction + SHAP drivers + nearest
  stations) and produces natural language — no external data enters the advisory. Falls
  back to a deterministic template if unreachable.

## Demo Scenarios (preloaded in sidebar)
- VIP Movement — Bellary Road
- Public Rally — Mysore Road (HIGH impact)
- Construction — ORR North
- Truck Breakdown — Tumkur Road
- Procession — CBD
