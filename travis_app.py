"""
TRAVIS - Traffic Risk & Advisory Intelligence System
Bangalore Event-Driven Traffic Congestion Prediction
Phase 4: Streamlit Demo App
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
import folium
from streamlit_folium import st_folium
from datetime import datetime, date, time
import warnings
warnings.filterwarnings("ignore")

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
MODEL_DIR = "/models"

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
# FEATURE ENGINEERING  (mirrors Phase 2 exactly)
# ─────────────────────────────────────────────────────────────────────
def engineer_features(event_cause, corridor, police_station, event_dt,
                      event_type="unplanned"):
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

    # Junction defaults (no junction at input time)
    j_clos   = gm["junc_closure_rt"]
    j_hpri   = gm["junc_highprio_rt"]
    j_hs     = gm["junction_hotspot_score"]
    j_total  = gm["junc_total"]

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
    )
    st.session_state.result   = predict(X, event_cause_key=sel_cause)
    st.session_state.feat_vec = X

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
        st.markdown('<div class="section-title">📍 Impact Map — Bangalore</div>', unsafe_allow_html=True)

        coords = LOOKUP["corr_coords"].get(sel_corridor, {"lat": 12.9716, "lon": 77.5946})
        clat, clon = coords["lat"], coords["lon"]

        RADIUS = {"HIGH": 2000, "MEDIUM": 1200, "LOW": 600}
        OPACITY = {"HIGH": 0.35, "MEDIUM": 0.25, "LOW": 0.15}
        MAP_COLOR = {"HIGH":"#ff6b6b","MEDIUM":"#ffd166","LOW":"#00d4aa"}

        m = folium.Map(
            location=[clat, clon], zoom_start=13,
            tiles="CartoDB dark_matter",
        )

        # Impact radius circle
        folium.Circle(
            location=[clat, clon],
            radius=RADIUS[tier],
            color=MAP_COLOR[tier],
            fill=True,
            fill_color=MAP_COLOR[tier],
            fill_opacity=OPACITY[tier],
            weight=2,
            popup=f"Impact Zone: {RADIUS[tier]}m radius",
        ).add_to(m)

        # Event marker
        icon_html = f"""
        <div style="background:{MAP_COLOR[tier]};border-radius:50%;width:20px;height:20px;
                    border:3px solid white;box-shadow:0 0 10px {MAP_COLOR[tier]}88;"></div>
        """
        folium.Marker(
            location=[clat, clon],
            popup=folium.Popup(
                f"<b>{sel_cause.replace('_',' ').title()}</b><br>"
                f"Corridor: {sel_corridor}<br>"
                f"Impact: {impact:.0f}/100<br>"
                f"P(Closure): {res['p_closure']:.1f}%",
                max_width=200
            ),
            icon=folium.DivIcon(html=icon_html, icon_size=(20,20), icon_anchor=(10,10)),
        ).add_to(m)

        # Nearby police stations (top 3 by distance)
        for ps_name, ps_data in list(LOOKUP["police_station"].items())[:6]:
            if ps_name == sel_ps:
                folium.Marker(
                    location=[clat + np.random.uniform(-0.01, 0.01),
                              clon + np.random.uniform(-0.01, 0.01)],
                    popup=f"PS: {ps_name}",
                    icon=folium.Icon(color="blue", icon="shield", prefix="fa"),
                ).add_to(m)
                break

        st_folium(m, width=None, height=420, returned_objects=[])

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
