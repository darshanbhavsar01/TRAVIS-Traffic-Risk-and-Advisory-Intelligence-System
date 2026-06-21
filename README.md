# TRAVIS — Traffic Risk & Advisory Intelligence System

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Ensure this folder structure:
#    outputs/
#      models/
#        model_road_closure.pkl
#        model_priority.pkl
#        model_resolution.pkl
#        model_meta.json
#        lookup_tables.json
#      travis_app.py
#      requirements.txt

# 3. Run the app (from the outputs/ folder)
streamlit run travis_app.py
```

## What the App Does

| Feature | Detail |
|---------|--------|
| Impact Score | 0–100 composite from 3 ML models |
| Road Closure | LightGBM binary classifier (AUC 0.72) |
| Priority | LightGBM binary classifier (AUC 0.9995) |
| Duration | LightGBM regressor, log-transformed |
| SHAP | Per-prediction feature attribution |
| Map | Folium dark map with impact radius |
| Deployment Plan | Rule engine: officers, barricades, response time |

## Demo Scenarios (preloaded in sidebar)
- 🔴 VIP Movement — Bellary Road
- 🟠 Public Rally — Mysore Road (HIGH impact)
- 🟡 Construction — ORR North
- ⚪ Truck Breakdown — Tumkur Road
- 🟠 Procession — CBD
