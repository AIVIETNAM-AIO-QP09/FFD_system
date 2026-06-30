import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import json
import plotly.graph_objects as go

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from dashboard_utils import (
        load_artifacts, load_demo_transactions, get_transaction_by_id,
        score_transaction, build_key_risk_factors, retrieve_similar_cases,
        generate_guarded_brief_dict, check_lmstudio_status
    )
    from feedback_manager import get_pending_queue, submit_analyst_feedback, get_queue_stats
except ImportError as e:
    st.error(f"Failed to import modules: {e}")
    st.stop()

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FFD System — Fraud Analyst Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── GLOBAL CSS: DARK FINTECH GLASSMORPHISM ────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ─── HEADER ─── */
.ffd-header {
    background: linear-gradient(135deg, #0A0F1E 0%, #141929 50%, #1a2040 100%);
    border-bottom: 1px solid rgba(79,142,247,0.25);
    padding: 1rem 2rem;
    margin: -1rem -1rem 1.5rem -1rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.ffd-logo { font-size: 1.4rem; font-weight: 700; color: #E8EAF0; letter-spacing: -0.5px; }
.ffd-logo span { color: #4F8EF7; }
.ffd-live-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(16,185,129,0.15); border: 1px solid rgba(16,185,129,0.4);
    color: #10B981; padding: 4px 12px; border-radius: 20px; font-size: 0.75rem; font-weight: 600;
}
.ffd-live-dot { width:7px; height:7px; border-radius:50%; background:#10B981; animation: pulse 2s infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }

/* ─── KPI CARDS ─── */
.kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 1.5rem; }
.kpi-card {
    background: rgba(20,25,41,0.8);
    border: 1px solid rgba(79,142,247,0.2);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    backdrop-filter: blur(10px);
    transition: border-color 0.2s, transform 0.2s;
}
.kpi-card:hover { border-color: rgba(79,142,247,0.5); transform: translateY(-2px); }
.kpi-label { font-size: 0.72rem; font-weight: 500; color: #7C85A0; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 0.4rem; }
.kpi-value { font-size: 2rem; font-weight: 700; color: #E8EAF0; line-height: 1; }
.kpi-value.danger { color: #F87171; }
.kpi-value.warning { color: #FBBF24; }
.kpi-value.success { color: #34D399; }
.kpi-value.info { color: #4F8EF7; }
.kpi-sub { font-size: 0.72rem; color: #5A6480; margin-top: 0.3rem; }

/* ─── GLASS CARD ─── */
.glass-card {
    background: rgba(20,25,41,0.85);
    border: 1px solid rgba(79,142,247,0.15);
    border-radius: 14px;
    padding: 1.5rem;
    backdrop-filter: blur(12px);
    margin-bottom: 1rem;
}
.card-title {
    font-size: 0.8rem; font-weight: 600; color: #7C85A0;
    text-transform: uppercase; letter-spacing: 1px; margin-bottom: 1rem;
    padding-bottom: 0.6rem; border-bottom: 1px solid rgba(79,142,247,0.1);
}

/* ─── RISK BADGE ─── */
.risk-badge {
    display: inline-block; padding: 4px 14px; border-radius: 20px;
    font-size: 0.78rem; font-weight: 700; letter-spacing: 0.5px;
}
.risk-high   { background: rgba(248,113,113,0.15); border: 1px solid rgba(248,113,113,0.5); color: #F87171; }
.risk-review { background: rgba(251,191,36,0.15);  border: 1px solid rgba(251,191,36,0.5);  color: #FBBF24; }
.risk-medium { background: rgba(251,146,60,0.15);  border: 1px solid rgba(251,146,60,0.5);  color: #FB923C; }
.risk-low    { background: rgba(52,211,153,0.15);  border: 1px solid rgba(52,211,153,0.5);  color: #34D399; }

/* ─── TX DETAIL GRID ─── */
.tx-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.75rem; }
.tx-field { background: rgba(10,15,30,0.6); border-radius: 8px; padding: 0.7rem 1rem; }
.tx-field-label { font-size: 0.68rem; color: #5A6480; text-transform: uppercase; letter-spacing: 0.6px; margin-bottom: 0.2rem; }
.tx-field-value { font-size: 0.9rem; color: #C8CCDC; font-weight: 500; }

/* ─── BRIEF SECTIONS ─── */
.brief-section { border-radius: 10px; padding: 1rem 1.2rem; margin-bottom: 0.75rem; }
.brief-explain { background: rgba(79,142,247,0.08); border-left: 3px solid #4F8EF7; }
.brief-steps   { background: rgba(251,191,36,0.08); border-left: 3px solid #FBBF24; }
.brief-caveats { background: rgba(248,113,113,0.08); border-left: 3px solid #F87171; }
.brief-label { font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 0.5rem; }
.brief-text { font-size: 0.88rem; color: #B0B8CC; line-height: 1.6; }

/* ─── DECISION BUTTONS ─── */
.decision-bar {
    background: rgba(10,15,30,0.9);
    border: 1px solid rgba(79,142,247,0.15);
    border-radius: 14px; padding: 1.5rem;
    margin-top: 1rem;
}
.decision-title { font-size: 0.78rem; font-weight: 600; color: #7C85A0; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 1rem; }

/* ─── SIDEBAR QUEUE ITEM ─── */
.queue-item {
    background: rgba(20,25,41,0.7);
    border: 1px solid rgba(79,142,247,0.1);
    border-radius: 8px; padding: 0.6rem 0.8rem; margin-bottom: 0.4rem;
    cursor: pointer; transition: border-color 0.15s;
}
.queue-item:hover { border-color: rgba(79,142,247,0.4); }
.queue-item.selected { border-color: #4F8EF7; background: rgba(79,142,247,0.1); }

/* ─── OVERRIDE STREAMLIT DEFAULTS ─── */
div[data-testid="stSidebar"] { background: #0D1220 !important; border-right: 1px solid rgba(79,142,247,0.15); }
.stButton > button {
    border-radius: 8px !important; font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important; transition: all 0.2s !important;
}
div[data-testid="stMetric"] { background: rgba(20,25,41,0.6) !important; border-radius: 10px !important; padding: 0.8rem !important; }
</style>
""", unsafe_allow_html=True)

# ── LOAD ARTIFACTS ───────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_artifacts():
    return load_artifacts()

@st.cache_data(show_spinner=False)
def get_demo_data():
    return load_demo_transactions()

with st.spinner("Initializing FFD System..."):
    preprocessor, model, retriever, corpus = get_artifacts()
    demo_df = get_demo_data()

lm_online = check_lmstudio_status()

# ── HEADER ───────────────────────────────────────────────────────────────────
q_stats = get_queue_stats()
pending_count = q_stats.get('PENDING', 0)
fb_count = q_stats.get('TOTAL_FEEDBACK', 0)
lm_status = "🟢 LM Studio Online" if lm_online else "🟡 LM Studio Offline"

st.markdown(f"""
<div class="ffd-header">
    <div>
        <div class="ffd-logo">🛡️ <span>FFD</span> System</div>
        <div style="font-size:0.72rem; color:#5A6480; margin-top:2px;">Financial Fraud Detection & Review Co-pilot</div>
    </div>
    <div style="display:flex; gap:12px; align-items:center;">
        <div style="font-size:0.75rem; color:#7C85A0;">{lm_status}</div>
        <div class="ffd-live-badge"><div class="ffd-live-dot"></div> LIVE STREAM ACTIVE</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── KPI ROW ──────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="kpi-grid">
    <div class="kpi-card">
        <div class="kpi-label">📋 Pending Review</div>
        <div class="kpi-value warning">{pending_count}</div>
        <div class="kpi-sub">Transactions in queue</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">✅ Labeled Today</div>
        <div class="kpi-value success">{fb_count}</div>
        <div class="kpi-sub">Human decisions recorded</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">🛑 Auto-Blocked</div>
        <div class="kpi-value danger">0</div>
        <div class="kpi-sub">Score ≥ 0.80 auto-rejected</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">🤖 Model</div>
        <div class="kpi-value info" style="font-size:1.1rem; padding-top:0.3rem;">LightGBM v1</div>
        <div class="kpi-sub">ROC-AUC: 0.623 · PR-AUC: 0.048</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── GUARD ────────────────────────────────────────────────────────────────────
if demo_df.empty or not preprocessor:
    st.error("❌ Model artifacts or demo data missing. Please check your `models/` directory.")
    st.stop()

# ── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Control Panel")

    with st.expander("⚠️ Safety Guidelines", expanded=False):
        st.caption(
            "• Model provides a ranking signal only.\n"
            "• LLM does NOT make fraud decisions.\n"
            "• Do NOT use for automatic blocking.\n"
            "• Human analyst has final authority."
        )

    st.markdown("---")
    data_mode = st.radio(
        "📡 Data Source",
        ["🔴 Live Review Queue", "📁 Static Demo"],
        help="Live Queue = transactions flagged by real-time engine. Static = 8 demo transactions."
    )

    if "Live" in data_mode:
        pending_df = get_pending_queue(200)
        if pending_df.empty:
            st.warning("Queue empty. Using Static Demo.")
            active_df = demo_df
        else:
            active_df = pending_df
        st.caption(f"**{pending_count}** pending · **{fb_count}** labeled")
    else:
        active_df = demo_df
        st.caption(f"**{len(demo_df)}** demo transactions loaded")

    st.markdown("---")
    tx_ids = active_df['transaction_id'].tolist()
    selected_tx = st.selectbox("🔍 Transaction ID", tx_ids)

    st.markdown("---")
    use_llm = st.toggle("Use LM Studio (LLM)", value=True)
    show_tech = st.toggle("Show Technical Details", value=False)

    st.markdown("---")
    analyze_btn = st.button("⚡ Generate Analyst Brief", type="primary", use_container_width=True)

    st.markdown("---")
    st.markdown("<div style='font-size:0.65rem; color:#3A4060; text-align:center;'>FFD System v2.0 · Active Learning</div>", unsafe_allow_html=True)

# ── MAIN AREA ─────────────────────────────────────────────────────────────────
if not analyze_btn and 'analyzed_tx' not in st.session_state:
    st.markdown("""
    <div class="glass-card" style="text-align:center; padding: 3rem 2rem;">
        <div style="font-size:3rem; margin-bottom:1rem;">🛡️</div>
        <div style="font-size:1.3rem; font-weight:600; color:#C8CCDC; margin-bottom:0.5rem;">
            Welcome to FFD Analyst Dashboard
        </div>
        <div style="font-size:0.9rem; color:#5A6480; max-width:480px; margin:0 auto; line-height:1.7;">
            Select a <strong style="color:#4F8EF7;">Data Source Mode</strong> in the sidebar, 
            choose a <strong style="color:#4F8EF7;">Transaction ID</strong> from the queue, 
            then click <strong style="color:#4F8EF7;">Generate Analyst Brief</strong> to begin your investigation.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if "Live" in data_mode and not active_df.empty:
        st.markdown('<div class="card-title" style="margin-top:1rem;">📋 Live Queue Preview (Top 15 by Risk)</div>', unsafe_allow_html=True)
        preview_cols = [c for c in ['transaction_id','amount','risk_score','risk_band','transaction_type','payment_channel','location'] if c in active_df.columns]
        st.dataframe(
            active_df[preview_cols].head(15).style.format({
                'amount': '${:,.2f}',
                'risk_score': '{:.4f}'
            } if 'amount' in preview_cols and 'risk_score' in preview_cols else {}),
            use_container_width=True,
            hide_index=True
        )
    st.stop()

if analyze_btn:
    st.session_state['analyzed_tx'] = selected_tx

current_tx = st.session_state.get('analyzed_tx', selected_tx)
row = get_transaction_by_id(active_df, current_tx)
if row is None:
    row = get_transaction_by_id(demo_df, current_tx)
if row is None:
    st.error(f"Transaction `{current_tx}` not found.")
    st.stop()

with st.spinner("Scoring transaction & generating AI brief..."):
    score, risk_band = score_transaction(row, preprocessor, model)
    risk_factors = build_key_risk_factors(row, score, risk_band)
    sim_cases = retrieve_similar_cases(row, retriever, corpus, preprocessor)
    brief_data = generate_guarded_brief_dict(current_tx, score, risk_band, risk_factors, sim_cases, use_llm=use_llm)

# Helper: risk class
def risk_class(band):
    return {'High':'risk-high','Review':'risk-review','Medium':'risk-medium','Low':'risk-low'}.get(band,'risk-low')

row_dict = {k: v for k, v in row.items()} if hasattr(row, 'items') else dict(row)

# ── TRANSACTION HEADER ────────────────────────────────────────────────────────
col_head1, col_head2, col_head3, col_head4 = st.columns([2,1,1,1])
with col_head1:
    st.markdown(f"""
    <div class="glass-card" style="padding:1rem 1.5rem;">
        <div class="kpi-label">Transaction ID</div>
        <div style="font-size:1.5rem; font-weight:700; color:#E8EAF0; font-family:'Roboto Mono', monospace;">{current_tx}</div>
        <div style="margin-top:0.5rem;">
            <span class="risk-badge {risk_class(risk_band)}">{risk_band.upper()} RISK</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
with col_head2:
    amt = float(row_dict.get('amount', 0))
    st.markdown(f"""
    <div class="glass-card" style="padding:1rem 1.5rem; text-align:center;">
        <div class="kpi-label">Amount</div>
        <div class="kpi-value" style="font-size:1.4rem;">${amt:,.2f}</div>
    </div>
    """, unsafe_allow_html=True)
with col_head3:
    st.markdown(f"""
    <div class="glass-card" style="padding:1rem 1.5rem; text-align:center;">
        <div class="kpi-label">Risk Score</div>
        <div class="kpi-value {'danger' if risk_band=='High' else 'warning' if risk_band=='Review' else 'success'}" style="font-size:1.4rem;">{score:.4f}</div>
    </div>
    """, unsafe_allow_html=True)
with col_head4:
    action_map = {'High':'Escalate to Senior','Review':'Manual Review','Medium':'Monitor','Low':'Approve'}
    st.markdown(f"""
    <div class="glass-card" style="padding:1rem 1.5rem; text-align:center;">
        <div class="kpi-label">Recommended</div>
        <div style="font-size:0.85rem; font-weight:600; color:#C8CCDC; margin-top:0.3rem;">{action_map.get(risk_band,'Approve')}</div>
    </div>
    """, unsafe_allow_html=True)

# ── RISK SCORE GAUGE ─────────────────────────────────────────────────────────
gauge_color = "#F87171" if risk_band == "High" else "#FBBF24" if risk_band in ("Review","Medium") else "#34D399"
fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=score,
    number={'font': {'size': 28, 'color': gauge_color, 'family': 'Inter'}, 'valueformat': '.4f'},
    gauge={
        'axis': {'range': [0, 0.2], 'tickcolor': '#5A6480', 'tickfont': {'color': '#5A6480', 'size': 10}},
        'bar': {'color': gauge_color, 'thickness': 0.25},
        'bgcolor': '#0A0F1E',
        'borderwidth': 0,
        'steps': [
            {'range': [0, 0.02],  'color': 'rgba(52,211,153,0.12)'},
            {'range': [0.02, 0.08], 'color': 'rgba(251,191,36,0.12)'},
            {'range': [0.08, 0.2],  'color': 'rgba(248,113,113,0.12)'},
        ],
        'threshold': {'line': {'color': gauge_color, 'width': 2}, 'thickness': 0.7, 'value': score}
    },
    domain={'x': [0, 1], 'y': [0, 1]}
))
fig_gauge.update_layout(
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    margin=dict(t=30, b=10, l=20, r=20), height=170
)

col_gauge, col_details = st.columns([1, 2])
with col_gauge:
    st.markdown('<div class="glass-card" style="padding:1rem;">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Risk Score Gauge</div>', unsafe_allow_html=True)
    st.plotly_chart(fig_gauge, use_container_width=True, config={'displayModeBar': False})
    st.markdown('</div>', unsafe_allow_html=True)

with col_details:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Transaction Details</div>', unsafe_allow_html=True)
    fields = [
        ("Transaction Type", row_dict.get('transaction_type','N/A')),
        ("Payment Channel",  row_dict.get('payment_channel','N/A')),
        ("Merchant Category",row_dict.get('merchant_category','N/A')),
        ("Device Used",      row_dict.get('device_used','N/A')),
        ("Location",         row_dict.get('location','N/A')),
        ("Timestamp",        str(row_dict.get('timestamp','N/A'))[:19]),
    ]
    grid_html = '<div class="tx-grid">'
    for label, value in fields:
        grid_html += f'<div class="tx-field"><div class="tx-field-label">{label}</div><div class="tx-field-value">{value}</div></div>'
    grid_html += '</div>'
    st.markdown(grid_html, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── TABS ──────────────────────────────────────────────────────────────────────
tab_labels = ["📋 Risk Factors", "🤖 AI Investigation Brief", "📊 Similar Cases"]
if show_tech:
    tab_labels.append("🔧 Technical")
tabs = st.tabs(tab_labels)

# ── TAB 1: RISK FACTORS ───────────────────────────────────────────────────────
with tabs[0]:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">⚑ Key Risk Signals Detected</div>', unsafe_allow_html=True)
    if risk_factors:
        for i, rf in enumerate(risk_factors, 1):
            st.markdown(f"""
            <div style="display:flex; gap:12px; align-items:flex-start; padding:0.6rem 0; border-bottom:1px solid rgba(79,142,247,0.08);">
                <div style="min-width:24px; height:24px; border-radius:6px; background:rgba(248,113,113,0.15); 
                     border:1px solid rgba(248,113,113,0.3); display:flex; align-items:center; justify-content:center;
                     font-size:0.7rem; color:#F87171; font-weight:700;">{i}</div>
                <div style="font-size:0.87rem; color:#B0B8CC; line-height:1.5; padding-top:2px;">{rf}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown('<div style="color:#5A6480; font-size:0.88rem;">No significant risk signals detected.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── TAB 2: AI BRIEF ───────────────────────────────────────────────────────────
with tabs[1]:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🤖 AI Investigation Brief</div>', unsafe_allow_html=True)
    
    guardrail_color = "#F87171" if "failed" in brief_data['status_msg'] else "#FBBF24" if "Offline" in brief_data['status_msg'] else "#34D399"
    st.markdown(f"""
    <div style="display:inline-flex; align-items:center; gap:8px; margin-bottom:1rem;
         background:rgba(20,25,41,0.8); border:1px solid {guardrail_color}33; 
         border-radius:8px; padding:6px 14px;">
        <div style="width:8px; height:8px; border-radius:50%; background:{guardrail_color};"></div>
        <span style="font-size:0.75rem; color:{guardrail_color};">{brief_data['status_msg']}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="brief-section brief-explain">
        <div class="brief-label" style="color:#4F8EF7;">📘 {brief_data['explanation_title']}</div>
        <div class="brief-text">{brief_data['explanation_text']}</div>
    </div>
    <div class="brief-section brief-steps">
        <div class="brief-label" style="color:#FBBF24;">📌 {brief_data['next_step_title']}</div>
        <div class="brief-text">{brief_data['next_step_text']}</div>
    </div>
    <div class="brief-section brief-caveats">
        <div class="brief-label" style="color:#F87171;">⚠️ {brief_data['caveats_title']}</div>
        <div class="brief-text">{brief_data['caveats_text']}</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── DECISION PANEL ────────────────────────────────────────────────────────
    st.markdown("""
    <div class="decision-bar">
        <div class="decision-title">🧑‍⚖️ Human-in-the-Loop Decision Panel · Active Learning</div>
    </div>
    """, unsafe_allow_html=True)
    
    notes = st.text_area(
        "📝 Analyst Notes",
        placeholder="Optional: Add investigation notes before making your decision...",
        key=f"notes_{current_tx}",
        height=80
    )
    
    col_d1, col_d2, col_d3 = st.columns(3)
    with col_d1:
        if st.button("🚨 Confirm Fraud", use_container_width=True, type="primary", key=f"fraud_{current_tx}"):
            submit_analyst_feedback(current_tx, "CONFIRM_FRAUD", 1, notes, json.dumps(row_dict, default=str))
            st.error(f"🚨 TX **{current_tx}** marked as **CONFIRMED FRAUD** and added to Active Learning pool.")
    with col_d2:
        if st.button("✅ False Alarm", use_container_width=True, key=f"legit_{current_tx}"):
            submit_analyst_feedback(current_tx, "FALSE_ALARM", 0, notes, json.dumps(row_dict, default=str))
            st.success(f"✅ TX **{current_tx}** marked as **LEGITIMATE** and added to Active Learning pool.")
    with col_d3:
        if st.button("🔍 Escalate", use_container_width=True, key=f"esc_{current_tx}"):
            submit_analyst_feedback(current_tx, "ESCALATE", -1, notes, json.dumps(row_dict, default=str))
            st.info(f"🔍 TX **{current_tx}** escalated to senior audit queue.")

# ── TAB 3: SIMILAR CASES ──────────────────────────────────────────────────────
with tabs[2]:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🗃️ Similar Historical Cases (Nearest Neighbor)</div>', unsafe_allow_html=True)
    st.caption("These are structurally similar past cases for reference — not proof of fraud.")
    if sim_cases:
        st.dataframe(pd.DataFrame(sim_cases), use_container_width=True, hide_index=True)
    else:
        st.markdown('<div style="color:#5A6480;">No similar cases found in retrieval corpus.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── TAB 4: TECHNICAL ──────────────────────────────────────────────────────────
if show_tech:
    with tabs[3]:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🔧 Technical Metadata</div>', unsafe_allow_html=True)
        st.json({
            "transaction_id": current_tx,
            "risk_score": float(score),
            "risk_band": risk_band,
            "model": "LightGBM FS1",
            "similar_cases_count": len(sim_cases),
            "lmstudio_online": lm_online,
            "llm_status": brief_data['status_msg']
        })
        st.markdown('<div class="card-title" style="margin-top:1rem;">Raw Row Data</div>', unsafe_allow_html=True)

        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.bool_):   return bool(obj)
                if isinstance(obj, np.integer): return int(obj)
                if isinstance(obj, np.floating): return float(obj)
                if isinstance(obj, np.ndarray): return obj.tolist()
                if isinstance(obj, pd.Timestamp): return str(obj)
                return super().default(obj)

        clean = {}
        for k, v in row_dict.items():
            try:
                clean[k] = None if pd.isna(v) else v
            except Exception:
                clean[k] = str(v)
        st.json(json.dumps(clean, cls=NpEncoder))
        st.markdown('</div>', unsafe_allow_html=True)
