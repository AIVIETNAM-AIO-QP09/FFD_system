import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import json
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
try:
    from dashboard_utils import (
        load_artifacts,
        load_demo_transactions,
        get_transaction_by_id,
        score_transaction,
        build_key_risk_factors,
        retrieve_similar_cases,
        generate_guarded_brief_dict,
        check_lmstudio_status
    )
except ImportError:
    st.error("Failed to import dashboard_utils. Make sure you run this from the project root.")
    st.stop()

st.set_page_config(page_title="Fraud Analyst Dashboard", layout="wide")

st.title("Fraud Analyst Dashboard — ML Ranking + LLM Review Co-pilot")
st.markdown("##### Weak but validated fraud-ranking baseline. Human-in-the-loop review only. No automatic blocking.")

# Sidebar Safety Notes
with st.sidebar.expander("Safety Notes", expanded=True):
    st.warning(
        "- The ML model is a weak ranking signal.\n"
        "- This dashboard is for manual review prioritization only.\n"
        "- The LLM does not make fraud decisions.\n"
        "- Similar cases are nearest-neighbor examples, not proof.\n"
        "- Do not use this system for automatic blocking, fund freezing, or account suspension."
    )

@st.cache_resource(show_spinner=False)
def get_artifacts():
    return load_artifacts()

@st.cache_data(show_spinner=False)
def get_demo_data():
    return load_demo_transactions()

with st.spinner("Loading demo data and model artifacts..."):
    preprocessor, model, retriever, corpus = get_artifacts()
    demo_df = get_demo_data()

lm_online = check_lmstudio_status()

# Sidebar Status
st.sidebar.markdown("### System Status")
if preprocessor and model and retriever:
    st.sidebar.success("✅ Model Artifacts: Loaded")
else:
    st.sidebar.error("❌ Model Artifacts: Missing")

if lm_online:
    st.sidebar.success("✅ LM Studio: Reachable")
else:
    st.sidebar.warning("⚠️ LM Studio: Not Reachable")

if demo_df.empty:
    st.sidebar.error("❌ Data: Missing")
    st.stop()
elif not preprocessor:
    st.stop()

st.sidebar.markdown("---")
tx_ids = demo_df['transaction_id'].tolist()
selected_tx = st.sidebar.selectbox("Select Transaction ID", tx_ids)

use_llm = st.sidebar.checkbox("Use LM Studio if available", value=True)
show_tech = st.sidebar.checkbox("Show technical details", value=False)

if st.sidebar.button("Generate Analyst Brief", type="primary"):
    row = get_transaction_by_id(demo_df, selected_tx)
    
    score, risk_band = score_transaction(row, preprocessor, model)
    risk_factors = build_key_risk_factors(row, score, risk_band)
    sim_cases = retrieve_similar_cases(row, retriever, corpus, preprocessor)
    brief_data = generate_guarded_brief_dict(selected_tx, score, risk_band, risk_factors, sim_cases, use_llm=use_llm)
    
    # Tabs layout
    tab_names = ["Overview", "Investigation", "Analyst Brief", "System Architecture"]
    if show_tech:
        tab_names.append("Technical Details")
    
    tabs = st.tabs(tab_names)
    
    # ------------------ OVERVIEW TAB ------------------
    with tabs[0]:
        st.markdown("### Project Positioning")
        st.info("Weak but validated fraud-ranking baseline | Human-in-the-loop review only | No automatic blocking")
        
        st.markdown("### Selected Transaction Summary")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Transaction ID", selected_tx)
        col2.metric("Risk Score", f"{score:.4f}")
        
        badge_color = "green"
        if risk_band == "High":
            badge_color = "red"
        elif risk_band == "Review":
            badge_color = "orange"
        elif risk_band == "Medium":
            badge_color = "#D4AF37" # Dark yellow
            
        col3.markdown(f"**Review Priority:** <br><span style='color:white; background-color:{badge_color}; padding:4px 8px; border-radius:4px; font-weight:bold;'>{risk_band}</span>", unsafe_allow_html=True)
        
        rec_action = "Approve"
        if risk_band == "High":
            rec_action = "Escalate to senior analyst"
        elif risk_band == "Review":
            rec_action = "Manual review"
        elif risk_band == "Medium":
            rec_action = "Monitor"
            
        col4.markdown(f"**Recommended Action:** <br><span>{rec_action}</span>", unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### Final Model Info")
        st.markdown("**Model:** LightGBM FS1")
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("PR-AUC", "0.04819")
        m_col2.metric("ROC-AUC", "0.62329")
        m_col3.metric("Precision@1%", "4.78%")
        m_col4.metric("Precision@5%", "4.94%")
        
        st.markdown("---")
        st.markdown("### LLM Status")
        if lm_online and use_llm:
            st.success("Online: Qwen3-8B candidate wording + guardrail")
        else:
            st.warning("Offline: deterministic fallback mode")
            
    # ------------------ INVESTIGATION TAB ------------------
    with tabs[1]:
        st.markdown("### Transaction Details")
        details = {
            "Amount": f"${row['amount']:.2f}",
            "Transaction Type": str(row['transaction_type']),
            "Payment Channel": str(row['payment_channel']),
            "Merchant Category": str(row['merchant_category']),
            "Device Used": str(row['device_used']),
            "Location": str(row['location']),
            "Timestamp": str(row['timestamp']),
            "Hour": str(row['hour']),
            "TSLT Missing": "Yes" if row['tslt_is_missing'] else "No"
        }
        st.table(pd.DataFrame(list(details.items()), columns=["Field", "Value"]))
        
        st.markdown("### Key Risk Factors")
        for rf in risk_factors:
            st.markdown(f"- {rf}")
            
        st.markdown("### Similar Historical Cases")
        st.caption("Similar cases are structured nearest-neighbor examples, not causal proof of fraud.")
        st.dataframe(pd.DataFrame(sim_cases), use_container_width=True)

    # ------------------ ANALYST BRIEF TAB ------------------
    with tabs[2]:
        st.markdown("### Fraud Investigation Brief")
        
        st.info(f"**{brief_data['explanation_title']}**\n\n{brief_data['explanation_text']}")
        st.warning(f"**{brief_data['next_step_title']}**\n\n{brief_data['next_step_text']}")
        st.error(f"**{brief_data['caveats_title']}**\n\n{brief_data['caveats_text']}")
        
        st.markdown("---")
        st.markdown("#### Guardrail Status")
        if "Offline" in brief_data['status_msg']:
            st.warning(brief_data['status_msg'])
        elif "failed" in brief_data['status_msg']:
            st.error(brief_data['status_msg'])
        else:
            st.success(brief_data['status_msg'])

    # ------------------ SYSTEM ARCHITECTURE TAB ------------------
    with tabs[3]:
        st.markdown("### System Architecture")
        
        # Check if diagram exists
        img_path = "docs/assets/system_architecture.png"
        if os.path.exists(img_path):
            st.image(img_path, use_column_width=True)
        else:
            st.markdown("""
```text
Raw transaction data
→ Timestamp split
→ EDA & leakage audit
→ Feature engineering
→ LightGBM fraud-ranking model
→ Risk score / risk band
→ Evidence builder
→ Similar-case retriever
→ Qwen3-8B candidate wording
→ Validator-first deterministic guardrail
→ Analyst-facing fraud investigation brief
```
""")
        st.markdown("""
#### The Four Layers of the System
1. **Data & Research:** Raw transaction ingestion, timestamp-based holdout splitting, EDA, and strict leakage control.
2. **ML Fraud Ranking Pipeline:** Feature engineering (Baseline + strictly past Novelty binary) feeding into a tuned LightGBM classifier.
3. **LLM/RAG Fraud Review Co-pilot:** The investigation workflow. A nearest-neighbor retriever fetches similar structured cases, and a local Qwen3-8B proposes candidate wording. 
4. **Governance & Monitoring:** A deterministic guardrail layer strictly validates all LLM outputs to prevent unsupported risk language or automated blocking directives.
""")

    # ------------------ TECHNICAL DETAILS TAB ------------------
    if show_tech:
        with tabs[4]:
            st.markdown("### Technical Context")
            
            st.json({
                "model_version": "LightGBM FS1 Baseline",
                "features_used": len(row.to_dict()) - 3, # Approximate base features
                "similar_cases_count": len(sim_cases),
                "lmstudio_reachable": lm_online,
                "use_llm_flag": use_llm,
                "llm_fallback_status": brief_data['status_msg']
            })
            
            st.markdown("#### Selected Transaction Raw JSON")
            # Drop some heavy non-serializable stuff safely
            clean_row = {k: str(v) if pd.isna(v) else v for k, v in row.to_dict().items()}
            # To handle numpy types in json
            class NpEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.integer): return int(obj)
                    if isinstance(obj, np.floating): return float(obj)
                    if isinstance(obj, np.ndarray): return obj.tolist()
                    if isinstance(obj, pd.Timestamp): return str(obj)
                    return super(NpEncoder, self).default(obj)
                    
            st.json(json.dumps(clean_row, cls=NpEncoder))
            
            st.markdown("#### Risk Factors (Evidence Builder)")
            st.json(risk_factors)
