# TRAVIS — Architecture & Flow Diagrams

Diagrams for the **Traffic Risk & Advisory Intelligence System** (Bangalore event-driven
traffic congestion prediction). Rendered from the actual codebase: 3 LightGBM models +
32 engineered features (`model_meta.json`), 6 Mappls integration points, a Groq
natural-language advisory, and a printable public bulletin.

---

## 1. System Architecture

```mermaid
graph TB
    subgraph USER["User Layer"]
        OFF["Traffic Officer / Control Room"]
        CIT["City Commuters"]
    end

    subgraph APP["Streamlit App — travis_app.py"]
        SB["Sidebar Inputs<br/>corridor · cause · PS · junction<br/>day · hour · planned/unplanned"]
        FE["engineer_features()<br/>builds 32-feature vector"]
        PRED["predict()<br/>impact score + tier"]
        SHAP["get_shap_values()<br/>top risk drivers"]
        UI["Result UI<br/>gauge · metrics · map · cards"]
        FACTS["Assemble model-only<br/>facts JSON"]
        BULL["build_bulletin_html()<br/>printable advisory"]
    end

    subgraph ML["ML / Data Assets — models/"]
        M1["model_road_closure.pkl<br/>LightGBM · P(closure)"]
        M2["model_priority.pkl<br/>LightGBM · P(high priority)"]
        M3["model_resolution.pkl<br/>LightGBM · duration hrs"]
        LU["lookup_tables.json<br/>corridor/PS/cause/zone/junction stats"]
        META["model_meta.json<br/>features · thresholds · weights"]
        JC["junction_coords.json<br/>(precomputed geocodes)"]
    end

    subgraph EXT["External APIs"]
        subgraph MAP["Mappls (Map My India)"]
            TOK["OAuth Token<br/>outpost.mappls.com"]
            SDK["Map JS SDK + Traffic Layer"]
            RT["Route API<br/>diversion + ETA"]
            DM["Distance Matrix API<br/>nearest PS by road"]
            GC["Geocode API<br/>junction → lat/lon"]
            SM["Static Map API<br/>still_image snapshot"]
        end
        GROQ["Groq LLM<br/>llama-3.3-70b<br/>NL advisory (model data only)"]
    end

    PRE["precompute_junction_coords.py<br/>one-time batch geocode"]

    OFF --> SB
    SB --> FE
    LU --> FE
    META --> FE
    JC --> FE
    FE --> PRED
    M1 --> PRED
    M2 --> PRED
    M3 --> PRED
    PRED --> UI
    PRED --> SHAP
    M1 --> SHAP
    SHAP --> UI

    TOK --> SDK & RT & DM & GC & SM
    GC --> FE
    SDK --> UI
    RT --> UI
    DM --> UI

    PRED --> FACTS
    SHAP --> FACTS
    DM --> FACTS
    FACTS --> GROQ
    GROQ --> UI
    FACTS --> BULL
    SM --> BULL
    BULL --> CIT
    UI --> OFF

    PRE --> GC
    PRE --> JC

    classDef ext fill:#2d1b4e,stroke:#6c63ff,color:#fff
    classDef ml fill:#1b3a2d,stroke:#00d4aa,color:#fff
    classDef app fill:#1a1d27,stroke:#ffd166,color:#fff
    class TOK,SDK,RT,DM,GC,SM,GROQ,MAP ext
    class M1,M2,M3,LU,META,JC ml
    class SB,FE,PRED,SHAP,UI,FACTS,BULL app
```

---

## 2. Prediction & Advisory Flow

```mermaid
flowchart TD
    START([Officer selects event scenario]) --> CLICK[Click 'Forecast Impact']
    CLICK --> ENG["engineer_features()<br/>look up corridor/PS/cause/junction stats<br/>+ cyclical time encodings → 32 features"]

    ENG --> P1["model_road_closure → P(closure)"]
    ENG --> P2["model_priority → P(high priority)"]
    ENG --> P3["model_resolution → duration (hrs)"]

    P1 --> SCORE["Impact Score<br/>= P_closure·55 + cause_rank·30 + res·15"]
    P3 --> SCORE
    SCORE --> TIER{Impact tier?}
    TIER -->|≥50| HIGH[HIGH RISK]
    TIER -->|25–49| MED[MEDIUM RISK]
    TIER -->|<25| LOW[LOW RISK]

    HIGH & MED & LOW --> RULES["Deployment plan<br/>constables · barricades · response min"]

    CLICK --> MAPPLS{Mappls token<br/>available?}
    MAPPLS -->|yes| GEO["Resolve junction pin<br/>precomputed → live geocode → centroid"]
    GEO --> NEAR["Distance Matrix<br/>nearest police stations"]
    GEO --> ROUTE["Route API<br/>blocked vs diversion polyline + ETA"]
    GEO --> MAPUI["Render Mappls map<br/>SDK + traffic layer + markers"]
    MAPPLS -->|no| FALLBACK["Graceful fallback<br/>synthetic route / straight-line dist"]

    RULES --> DIV["Commuter Advisory<br/>blocked ETA × congestion(impact)<br/>vs diversion ETA → divert/stay"]
    ROUTE --> DIV
    P1 --> SH["SHAP top-4 risk drivers"]

    RULES --> JSON["Assemble facts JSON<br/>(model output + SHAP + nearest PS only)"]
    SH --> JSON
    NEAR --> JSON
    DIV --> JSON

    JSON --> GROQ{Groq reachable?}
    GROQ -->|yes| NL["3-sentence NL advisory<br/>(no external info)"]
    GROQ -->|no| TMPL["Deterministic template<br/>from same facts"]

    NL & TMPL --> ADVUI[Show AI Advisory card]
    JSON --> PDF["build_bulletin_html()<br/>+ static map snapshot"]
    ADVUI --> PDF
    PDF --> DL[["⬇ Download Public Bulletin<br/>HTML → print to PDF"]]
    DL --> CITIZENS([Posted online for commuters])
    MAPUI --> DASH([Officer dashboard])
    DIV --> DASH
```

---

## Component Notes

- **Three independent LightGBM models** feed one weighted **impact score**
  (55 / 30 / 15 weights from `model_meta.json`), which drives the risk tier and the
  rule-based deployment plan.
- **Mappls is used in 6 places** — OAuth token, JS SDK + live traffic layer, Route API,
  Distance Matrix API, Geocode API, and Static Map API. Every call has a graceful
  fallback (synthetic route / straight-line distance / corridor centroid) so the demo
  never breaks offline.
- **The Groq boundary is strict**: it only ever receives the assembled facts JSON
  (model outputs + SHAP drivers + nearest stations) and only produces prose — no
  external data enters the advisory.
- `junction_coords.json` is produced one-time by `precompute_junction_coords.py`; until
  it exists the app falls back to live geocoding → corridor centroid.
