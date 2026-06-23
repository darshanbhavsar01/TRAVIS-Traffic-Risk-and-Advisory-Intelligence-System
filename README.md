# TRAVIS — Traffic Risk & Advisory Intelligence System

**Live App:** https://travis-traffic-risk-and-advisory-intelligence-system-6fy3zvxqh.streamlit.app/

> **Architecture & flow diagrams:** see **[DIAGRAMS.md](DIAGRAMS.md)**.

## Description

Traffic disruption in Bangalore is rarely random — it is driven by events: VIP movements,
rallies, construction, breakdowns, processions. TRAVIS turns those events into actionable
foresight. Given an event (corridor, cause, time, location), it predicts **how likely a road
closure is, how high-priority the response should be, and how long the disruption will last**,
combining all three into a single 0–100 impact score.

From that prediction it generates a complete operational picture:

- A **resource deployment plan** (constables, barricades, target response time).
- An **interactive Mappls map** with lane-level data, a live traffic layer, the impact radius,
  the nearest police stations by road distance, and a **diversion route** (blocked road in red,
  alternate in green) with side-by-side ETAs so commuters know whether to divert.
- A **SHAP explanation** showing exactly which factors drove the prediction.
- An **AI-written advisory** (Groq `llama-3.3-70b`) that phrases the model's output into a
  plain-language briefing for a duty officer — strictly from model data, nothing invented.
- A **one-click public traffic bulletin** (printable HTML/PDF) that authorities can post online
  for the public to follow.

The result bridges the gap between a machine-learning score and something an officer can act on
and a citizen can read.

## How to Run

The fastest way to test TRAVIS is the **[live app](https://travis-traffic-risk-and-advisory-intelligence-system-6fy3zvxqh.streamlit.app/)** —
no setup needed. Pick a preloaded scenario from the sidebar (or build your own) and click
**Forecast Impact**.

To run it locally:

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
