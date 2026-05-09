"""
SecureNet IDS — Streamlit Dashboard
====================================
Full dashboard: model switcher, live classifier, metrics, charts.

Run locally:
    streamlit run app.py

Deploy to Streamlit Cloud:
    Push this file + requirements.txt to your GitHub repo root.
    Go to share.streamlit.io → New app → select repo → app.py
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SecureNet IDS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS (dark cybersecurity theme) ─────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #050a0e; color: #c8e8f8; }

    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #0a1520; border-right: 1px solid #0e3a5c; }

    /* Metric cards */
    [data-testid="stMetric"] {
        background-color: #0a1520;
        border: 1px solid #0e3a5c;
        border-top: 2px solid #00d4ff;
        padding: 16px;
        border-radius: 0px;
    }
    [data-testid="stMetricLabel"] { color: #4a7a9b !important; font-family: monospace; font-size: 11px; letter-spacing: 2px; }
    [data-testid="stMetricValue"] { color: #00d4ff !important; font-family: monospace; }

    /* Headers */
    h1, h2, h3 { color: #00d4ff !important; font-family: monospace; letter-spacing: 2px; }
    h2 { color: #c8e8f8 !important; font-size: 16px !important; }

    /* Buttons */
    .stButton > button {
        background-color: #00d4ff;
        color: #050a0e;
        border: none;
        font-family: monospace;
        font-weight: bold;
        letter-spacing: 2px;
        border-radius: 0px;
        width: 100%;
    }
    .stButton > button:hover { background-color: #33ddff; box-shadow: 0 0 20px rgba(0,212,255,0.5); }

    /* Input fields */
    .stNumberInput input, .stTextInput input {
        background-color: #050a0e !important;
        border: 1px solid #0e3a5c !important;
        color: #c8e8f8 !important;
        font-family: monospace !important;
        border-radius: 0px !important;
    }

    /* Selectbox */
    .stSelectbox > div > div {
        background-color: #0a1520 !important;
        border: 1px solid #0e3a5c !important;
        border-radius: 0px !important;
        color: #c8e8f8 !important;
    }

    /* Divider */
    hr { border-color: #0e3a5c; }

    /* Success / error boxes */
    .attack-box {
        background: rgba(255,60,92,0.1);
        border: 2px solid #ff3c5c;
        border-left: 6px solid #ff3c5c;
        padding: 20px;
        font-family: monospace;
        font-size: 22px;
        font-weight: bold;
        color: #ff3c5c;
        letter-spacing: 3px;
        text-shadow: 0 0 20px rgba(255,60,92,0.5);
        margin: 12px 0;
    }
    .benign-box {
        background: rgba(0,255,157,0.07);
        border: 2px solid #00ff9d;
        border-left: 6px solid #00ff9d;
        padding: 20px;
        font-family: monospace;
        font-size: 22px;
        font-weight: bold;
        color: #00ff9d;
        letter-spacing: 3px;
        text-shadow: 0 0 20px rgba(0,255,157,0.4);
        margin: 12px 0;
    }
    .info-row {
        font-family: monospace;
        font-size: 12px;
        color: #4a7a9b;
        margin-top: 8px;
    }

    /* Dataframe */
    .stDataFrame { border: 1px solid #0e3a5c; }

    /* Radio buttons */
    .stRadio label { color: #c8e8f8 !important; font-family: monospace; }

    /* Hide Streamlit branding */
    #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Load model artifacts ───────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    base = os.path.dirname(__file__)

    # Try backend/ subfolder first, then same directory
    paths = [
        os.path.join(base, 'backend'),
        base,
        os.path.join(base, '..', 'backend')
    ]

    for p in paths:
        model_path = os.path.join(p, 'model.pkl')
        if os.path.exists(model_path):
            try:
                rf      = joblib.load(os.path.join(p, 'model.pkl'))
                dt      = joblib.load(os.path.join(p, 'dt_model.pkl'))
                scaler  = joblib.load(os.path.join(p, 'scaler.pkl'))
                le      = joblib.load(os.path.join(p, 'label_encoder.pkl'))
                feats   = joblib.load(os.path.join(p, 'feature_names.pkl'))
                metrics = joblib.load(os.path.join(p, 'metrics.pkl'))
                return {'rf': rf, 'dt': dt}, scaler, le, feats, metrics
            except Exception as e:
                continue

    return None, None, None, None, None


artifacts = load_artifacts()
models_dict, scaler, le, feature_names, saved_metrics = artifacts

MODELS_LOADED = models_dict is not None

# Fallback metrics (your actual results from the notebook)
FALLBACK_METRICS = {
    'random_forest': {'accuracy': 0.9999, 'precision': 0.9999, 'recall': 0.9999, 'f1': 0.9999},
    'decision_tree': {'accuracy': 0.9996, 'precision': 0.9996, 'recall': 0.9996, 'f1': 0.9996}
}
display_metrics = saved_metrics if saved_metrics else FALLBACK_METRICS


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ SecureNet IDS")
    st.markdown("<p style='font-family:monospace;font-size:10px;color:#4a7a9b;letter-spacing:2px;'>AI-POWERED INTRUSION DETECTION</p>", unsafe_allow_html=True)
    st.divider()

    # Model selector
    st.markdown("### ⚙️ Active Model")
    model_choice = st.radio(
        "Select ML Model",
        options=["Random Forest", "Decision Tree"],
        index=0,
        label_visibility="collapsed"
    )
    active_key = 'rf' if model_choice == 'Random Forest' else 'dt'
    metrics_key = 'random_forest' if active_key == 'rf' else 'decision_tree'

    st.divider()

    # Model metrics in sidebar
    m = display_metrics[metrics_key]
    st.markdown("### 📊 Model Metrics")
    st.metric("Accuracy",  f"{m['accuracy']*100:.2f}%")
    st.metric("Precision", f"{m['precision']*100:.2f}%")
    st.metric("Recall",    f"{m['recall']*100:.2f}%")
    st.metric("F1-Score",  f"{m['f1']*100:.2f}%")

    st.divider()

    # Status
    if MODELS_LOADED:
        st.success("✅ Models loaded")
    else:
        st.warning("⚠️ Demo mode\nRun notebook first to load real models.")

    st.markdown("<p style='font-family:monospace;font-size:9px;color:#4a7a9b;'>Dataset: CIC-IDS2017<br>Friday DDoS Afternoon</p>", unsafe_allow_html=True)


# ── Main header ───────────────────────────────────────────────────────────────
st.markdown("# 🛡️ SECURENET IDS")
st.markdown("<p style='font-family:monospace;font-size:11px;color:#4a7a9b;letter-spacing:3px;margin-top:-14px;'>AI-POWERED NETWORK INTRUSION DETECTION SYSTEM — CIC-IDS2017</p>", unsafe_allow_html=True)
st.divider()


# ── Top metrics row ────────────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)
m = display_metrics[metrics_key]
col1.metric("📦 Dataset",   "CIC-IDS2017")
col2.metric("🤖 Model",     model_choice)
col3.metric("🎯 Accuracy",  f"{m['accuracy']*100:.2f}%",  delta="Test Set")
col4.metric("📐 F1-Score",  f"{m['f1']*100:.2f}%",        delta="Weighted")
col5.metric("🔢 Features",  str(len(feature_names)) if feature_names else "78")

st.divider()


# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["⚡ Live Classifier", "📊 Model Comparison", "📈 Charts"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Live Classifier
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### ⚡ Live Traffic Classifier")
    st.markdown(f"<p style='font-family:monospace;font-size:11px;color:#4a7a9b;'>Active model: <b style='color:#00d4ff'>{model_choice}</b></p>", unsafe_allow_html=True)

    # Demo preset buttons
    c1, c2, c3 = st.columns([1, 1, 3])
    ddos_preset   = c1.button("🔴 Load DDoS Demo")
    benign_preset = c2.button("🟢 Load Benign Demo")

    # Default values
    if ddos_preset:
        st.session_state['dur']      = 1234567.0
        st.session_state['fwd_pkts'] = 850.0
        st.session_state['bwd_pkts'] = 0.0
        st.session_state['bytes_s']  = 982345.5
        st.session_state['pkts_s']   = 12000.3
        st.session_state['fwd_len']  = 46.2
        st.session_state['bwd_len']  = 0.0
        st.session_state['avg_pkt']  = 44.0
    elif benign_preset:
        st.session_state['dur']      = 5432100.0
        st.session_state['fwd_pkts'] = 12.0
        st.session_state['bwd_pkts'] = 10.0
        st.session_state['bytes_s']  = 1234.5
        st.session_state['pkts_s']   = 4.2
        st.session_state['fwd_len']  = 512.8
        st.session_state['bwd_len']  = 480.2
        st.session_state['avg_pkt']  = 496.5

    st.markdown("#### Network Flow Features")
    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
    r2c1, r2c2, r2c3, r2c4 = st.columns(4)

    dur      = r1c1.number_input("Flow Duration (µs)",       value=st.session_state.get('dur',      1234567.0), format="%.1f")
    fwd_pkts = r1c2.number_input("Total Fwd Packets",        value=st.session_state.get('fwd_pkts', 85.0),      format="%.0f")
    bwd_pkts = r1c3.number_input("Total Bwd Packets",        value=st.session_state.get('bwd_pkts', 0.0),       format="%.0f")
    bytes_s  = r1c4.number_input("Flow Bytes/s",             value=st.session_state.get('bytes_s',  98234.5),   format="%.2f")
    pkts_s   = r2c1.number_input("Flow Packets/s",           value=st.session_state.get('pkts_s',   1200.3),    format="%.2f")
    fwd_len  = r2c2.number_input("Fwd Packet Length Mean",   value=st.session_state.get('fwd_len',  46.2),      format="%.2f")
    bwd_len  = r2c3.number_input("Bwd Packet Length Mean",   value=st.session_state.get('bwd_len',  0.0),       format="%.2f")
    avg_pkt  = r2c4.number_input("Average Packet Size",      value=st.session_state.get('avg_pkt',  44.0),      format="%.2f")

    st.markdown("")
    classify_btn = st.button("▶ CLASSIFY TRAFFIC", use_container_width=False)

    if classify_btn:
        input_vals = {
            'Flow Duration':          dur,
            'Total Fwd Packets':      fwd_pkts,
            'Total Backward Packets': bwd_pkts,
            'Flow Bytes/s':           bytes_s,
            'Flow Packets/s':         pkts_s,
            'Fwd Packet Length Mean': fwd_len,
            'Bwd Packet Length Mean': bwd_len,
            'Average Packet Size':    avg_pkt,
        }

        if MODELS_LOADED:
            # Real prediction
            model = models_dict[active_key]
            try:
                values   = [float(input_vals.get(f, 0)) for f in feature_names]
                X        = np.array(values).reshape(1, -1)
                X_scaled = scaler.transform(X)
                pred     = model.predict(X_scaled)[0]
                probs    = model.predict_proba(X_scaled)[0]
                label    = le.inverse_transform([pred])[0]
                classes  = le.classes_.tolist()
                conf     = float(max(probs))
                prob_dict = {c: round(float(p), 4) for c, p in zip(classes, probs)}
                is_demo  = False
            except Exception as e:
                st.error(f"Prediction error: {e}")
                st.stop()
        else:
            # Demo simulation
            is_attack = pkts_s > 100 or fwd_pkts > 100
            label     = 'DDoS' if is_attack else 'BENIGN'
            conf      = 0.9999 if is_attack else 0.9996
            prob_dict = {'BENIGN': round(1-conf, 4), 'DDoS': round(conf, 4)} if is_attack else {'BENIGN': round(conf, 4), 'DDoS': round(1-conf, 4)}
            is_demo   = True

        is_attack = label != 'BENIGN'

        # Result display
        if is_attack:
            st.markdown(f'<div class="attack-box">⚠ DDoS ATTACK DETECTED</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="benign-box">✓ BENIGN TRAFFIC</div>', unsafe_allow_html=True)

        # Confidence bar
        conf_pct = conf * 100
        bar_color = "#ff3c5c" if is_attack else "#00ff9d"
        st.markdown(f"""
        <div style='margin:12px 0 4px;font-family:monospace;font-size:10px;color:#4a7a9b;letter-spacing:2px;'>CONFIDENCE LEVEL</div>
        <div style='background:#0d1e2e;border:1px solid #0e3a5c;height:10px;width:100%;'>
          <div style='background:{bar_color};height:100%;width:{conf_pct:.1f}%;transition:width 0.8s;'></div>
        </div>
        """, unsafe_allow_html=True)

        # Info row
        demo_tag  = " [DEMO MODE]" if is_demo else ""
        st.markdown(f"""
        <div class='info-row'>
          Confidence: <b style='color:#c8e8f8'>{conf_pct:.2f}%</b> &nbsp;|&nbsp;
          Model: <b style='color:#00d4ff'>{model_choice}{demo_tag}</b> &nbsp;|&nbsp;
          BENIGN: <b style='color:#c8e8f8'>{prob_dict.get('BENIGN',0)*100:.2f}%</b> &nbsp;|&nbsp;
          DDoS: <b style='color:#c8e8f8'>{prob_dict.get('DDoS',0)*100:.2f}%</b>
        </div>
        """, unsafe_allow_html=True)

        # Probability bar chart
        st.markdown("")
        fig, ax = plt.subplots(figsize=(5, 1.5))
        fig.patch.set_facecolor('#0a1520')
        ax.set_facecolor('#050a0e')
        classes_list = list(prob_dict.keys())
        vals_list    = list(prob_dict.values())
        colors       = ['#00ff9d' if c == 'BENIGN' else '#ff3c5c' for c in classes_list]
        ax.barh(classes_list, vals_list, color=colors, height=0.5)
        ax.set_xlim(0, 1)
        ax.tick_params(colors='#4a7a9b', labelsize=9)
        ax.spines[:].set_color('#0e3a5c')
        for spine in ax.spines.values(): spine.set_color('#0e3a5c')
        for i, v in enumerate(vals_list):
            ax.text(v + 0.01, i, f'{v*100:.2f}%', va='center', color='#c8e8f8', fontsize=9, fontfamily='monospace')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=False)
        plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Model Comparison
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 📊 Model Performance Comparison")

    # Metrics table
    rf_m = display_metrics['random_forest']
    dt_m = display_metrics['decision_tree']

    comparison_df = pd.DataFrame({
        'Metric':        ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Random Forest': [f"{rf_m['accuracy']*100:.2f}%",  f"{rf_m['precision']*100:.2f}%",
                          f"{rf_m['recall']*100:.2f}%",    f"{rf_m['f1']*100:.2f}%"],
        'Decision Tree': [f"{dt_m['accuracy']*100:.2f}%",  f"{dt_m['precision']*100:.2f}%",
                          f"{dt_m['recall']*100:.2f}%",    f"{dt_m['f1']*100:.2f}%"]
    })
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    st.divider()

    # Side by side metric cards
    st.markdown("#### Random Forest")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy",  f"{rf_m['accuracy']*100:.2f}%")
    c2.metric("Precision", f"{rf_m['precision']*100:.2f}%")
    c3.metric("Recall",    f"{rf_m['recall']*100:.2f}%")
    c4.metric("F1-Score",  f"{rf_m['f1']*100:.2f}%")

    st.markdown("#### Decision Tree")
    c1, c2, c3, c4 = st.columns(4)
    delta = lambda a, b: f"{(a-b)*100:+.2f}% vs RF"
    c1.metric("Accuracy",  f"{dt_m['accuracy']*100:.2f}%",  delta(dt_m['accuracy'],  rf_m['accuracy']))
    c2.metric("Precision", f"{dt_m['precision']*100:.2f}%", delta(dt_m['precision'], rf_m['precision']))
    c3.metric("Recall",    f"{dt_m['recall']*100:.2f}%",    delta(dt_m['recall'],    rf_m['recall']))
    c4.metric("F1-Score",  f"{dt_m['f1']*100:.2f}%",        delta(dt_m['f1'],        rf_m['f1']))

    st.divider()

    # Bar chart comparison
    st.markdown("#### Visual Comparison")
    metrics_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    rf_vals = [rf_m['accuracy'], rf_m['precision'], rf_m['recall'], rf_m['f1']]
    dt_vals = [dt_m['accuracy'], dt_m['precision'], dt_m['recall'], dt_m['f1']]

    x = np.arange(len(metrics_labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor('#0a1520')
    ax.set_facecolor('#050a0e')
    b1 = ax.bar(x - w/2, rf_vals, w, label='Random Forest', color='#00d4ff', edgecolor='#050a0e')
    b2 = ax.bar(x + w/2, dt_vals, w, label='Decision Tree',  color='#00ff9d', edgecolor='#050a0e')
    ax.set_ylim(0.995, 1.002)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_labels, color='#c8e8f8', fontfamily='monospace')
    ax.tick_params(colors='#4a7a9b')
    ax.spines[:].set_color('#0e3a5c')
    ax.legend(facecolor='#0a1520', edgecolor='#0e3a5c', labelcolor='#c8e8f8', fontsize=9)
    ax.set_title('Model Performance Comparison', color='#00d4ff', fontfamily='monospace', fontsize=12)
    for bar in [*b1, *b2]:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001,
                f'{bar.get_height()*100:.2f}%', ha='center', va='bottom',
                fontsize=8, color='#c8e8f8', fontfamily='monospace')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.divider()
    st.markdown("#### 🔐 Security Analysis")
    st.markdown("""
    <div style='font-family:monospace;font-size:12px;color:#c8e8f8;line-height:1.8;background:#0a1520;border:1px solid #0e3a5c;padding:20px;'>
    <b style='color:#00d4ff'>Why Random Forest is preferred:</b><br>
    → Ensemble of 100 trees reduces variance and prevents overfitting<br>
    → More robust to unseen traffic patterns in production<br>
    → Provides feature importance scores for analyst investigation<br><br>
    <b style='color:#00d4ff'>Security implication of high Recall:</b><br>
    → 99.99% recall = fewer than 1 in 10,000 DDoS flows missed<br>
    → False negatives (missed attacks) are the most dangerous outcome<br>
    → This model prioritises detection over false alarms<br><br>
    <b style='color:#ffaa00'>⚠ Limitation:</b><br>
    → CIC-IDS2017 is a controlled lab dataset — real-world traffic is noisier<br>
    → Model must be periodically retrained on new attack patterns
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Charts
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 📈 Analysis Charts")
    st.markdown("<p style='font-family:monospace;font-size:11px;color:#4a7a9b;'>Generated by running the Jupyter Notebook. Run all cells to produce these images.</p>", unsafe_allow_html=True)

    base = os.path.dirname(__file__)
    chart_dirs = [
        os.path.join(base, 'frontend'),
        os.path.join(base, '..', 'frontend'),
        base
    ]

    chart_files = {
        'Class Distribution':  'class_distribution.png',
        'Confusion Matrices':  'confusion_matrices.png',
        'Model Comparison':    'model_comparison.png',
        'Feature Importance':  'feature_importance.png',
        'Feature Distributions': 'feature_distributions.png'
    }

    # Find chart directory
    chart_dir = None
    for d in chart_dirs:
        if os.path.exists(os.path.join(d, 'class_distribution.png')):
            chart_dir = d
            break

    if chart_dir:
        c1, c2 = st.columns(2)
        items = list(chart_files.items())
        for i, (title, fname) in enumerate(items):
            path = os.path.join(chart_dir, fname)
            col = c1 if i % 2 == 0 else c2
            if os.path.exists(path):
                col.markdown(f"**{title}**")
                col.image(path, use_column_width=True)
    else:
        st.warning("⚠️ Charts not found. Run the Jupyter Notebook first — charts are auto-saved to the `frontend/` folder.")
        st.code("""
# Run this in your terminal:
cd notebook
jupyter notebook ids_model.ipynb
# Then: Cell → Run All
        """)
