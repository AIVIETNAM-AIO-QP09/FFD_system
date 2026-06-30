import os
import sys
import pickle
import pandas as pd
import numpy as np
import urllib.request
from openai import OpenAI

# Allow importing from src
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
try:
    from llm_client import validate_guardrails
except ImportError:
    # If llm_client is missing, define a mock validate_guardrails that always fails for safety
    def validate_guardrails(*args, **kwargs):
        return False, ["Missing llm_client"]

def get_lm_studio_url():
    return os.getenv("LM_STUDIO_URL", "http://127.0.0.1:1234/v1")

def check_lmstudio_status(base_url: str = None) -> bool:
    if base_url is None:
        base_url = get_lm_studio_url()
    try:
        req = urllib.request.Request(f"{base_url}/models", method="GET")
        with urllib.request.urlopen(req, timeout=2) as response:
            return response.status == 200
    except Exception:
        return False

def load_artifacts():
    try:
        with open("models/preprocessor.pkl", "rb") as f:
            preprocessor = pickle.load(f)
        with open("models/calibrated_lgbm.pkl", "rb") as f:
            model = pickle.load(f)
        with open("models/retriever.pkl", "rb") as f:
            retriever = pickle.load(f)
        corpus = pd.read_pickle("models/rag_corpus.pkl")
        return preprocessor, model, retriever, corpus
    except Exception as e:
        return None, None, None, None

def load_demo_transactions():
    try:
        if os.path.exists("data/demo_subset.pkl"):
            return pd.read_pickle("data/demo_subset.pkl")
            
        df = pd.read_csv("data/split/train.csv", parse_dates=['timestamp'])

        df = df.sort_values('timestamp').reset_index(drop=True)
        
        df['log_amount'] = np.log1p(df['amount'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)
        df['is_night'] = ((df['hour']>=0)&(df['hour']<=5)).astype(int)
        df['tslt_abs'] = df['time_since_last_transaction'].abs()
        df['tslt_is_missing'] = df['time_since_last_transaction'].isnull().astype(int)
        df['tslt_is_negative'] = (df['time_since_last_transaction']<0).astype(int)
        
        pairs = [
            ('sender_account', 'location', 'sender_location_pair_count_past', 'is_new_location_for_sender'),
            ('sender_account', 'payment_channel', 'sender_channel_pair_count_past', 'is_new_payment_channel_for_sender'),
            ('sender_account', 'device_used', 'sender_device_type_pair_count_past', 'is_new_device_used_for_sender'),
            ('sender_account', 'transaction_type', 'sender_txn_type_pair_count_past', 'is_new_transaction_type_for_sender'),
        ]
        for s, v, cnt_f, nov_f in pairs:
            df2 = df[[s, v, 'timestamp']].copy()
            df2['oi'] = df2.index
            df2 = df2.sort_values([s, v, 'timestamp'])
            df2['pc'] = df2.groupby([s, v]).cumcount()
            df2 = df2.sort_values('oi')
            df[nov_f] = (df2['pc'].values == 0).astype(int)
            
        target_ids = ["T2174172", "T4534855", "T1708346", "T1888153", "T2618028", "T3124730", "T2327555", "T124856"]
        subset = df[df['transaction_id'].isin(target_ids)].copy()
        
        if len(subset) == 0:
            subset = df.iloc[3200000:3200020].copy()
            
        subset.to_pickle("data/demo_subset.pkl")
        return subset
    except Exception as e:
        return pd.DataFrame()

def get_transaction_by_id(df, transaction_id):
    res = df[df['transaction_id'] == transaction_id]
    if len(res) == 0:
        return None
    return res.iloc[0]

def _safe_get_val(row, col, default):
    """Safely extract a single value from a pandas Series or dict, returning default if missing."""
    try:
        val = row[col]
        if val is None:
            return default
        # Handle numpy/pandas NA
        try:
            if pd.isna(val):
                return default
        except Exception:
            pass
        return val
    except (KeyError, IndexError, TypeError):
        return default

def score_transaction(row, preprocessor, model):
    NUM_HARD = ['amount','log_amount','spending_deviation_score','velocity_score',
               'geo_anomaly_score','hour','day_of_week','month','is_weekend','is_night',
               'tslt_abs','tslt_is_negative',
               'is_new_location_for_sender','is_new_payment_channel_for_sender',
               'is_new_transaction_type_for_sender','is_new_device_used_for_sender']
    CAT_COLS = ['transaction_type','merchant_category','payment_channel','device_used','location']
    
    # Build a BRAND NEW plain Python dict from scratch — never mutate the original row
    features = {}
    for col in NUM_HARD:
        raw = _safe_get_val(row, col, 0.0)
        try:
            features[col] = float(raw)
        except (ValueError, TypeError):
            features[col] = 0.0
    for col in CAT_COLS:
        raw = _safe_get_val(row, col, 'unknown')
        features[col] = str(raw) if raw is not None else 'unknown'
    
    df_row = pd.DataFrame([features])
    X = df_row[NUM_HARD + CAT_COLS]
    X_trans = preprocessor.transform(X)
    
    score = float(model.predict_proba(X_trans)[0, 1])
    
    # tslt_is_missing is optional (not present in Queue data) — safe default = 0
    if _safe_get_val(row, 'tslt_is_missing', 0) == 1:
        score = 1e-6
        
    if score >= 0.08:
        risk_band = "High"
    elif score >= 0.04:
        risk_band = "Review"
    elif score >= 0.01:
        risk_band = "Medium"
    else:
        risk_band = "Low"
        
    return score, risk_band


def build_key_risk_factors(row, score, risk_band):
    ev = []
    row_dict = row.to_dict() if hasattr(row, 'to_dict') else dict(row)
    if row_dict.get('tslt_is_missing', 0) == 1:
        ev.append("TSLT is missing; historically this group has near-zero fraud. This is treated as a low-risk structural bucket under the current model.")
    if row_dict.get('is_new_location_for_sender', 0) == 1:
        ev.append("Sender is using a new location for the first time.")
    if row.get('is_new_device_used_for_sender', 0) == 1:
        ev.append("Sender is using a new device type for the first time.")
    if row.get('is_new_payment_channel_for_sender', 0) == 1:
        ev.append("Sender is using a new payment channel.")
    if row.get('is_new_transaction_type_for_sender', 0) == 1:
        ev.append("Sender is using a new transaction type.")
    if row.get('is_night', 0) == 1:
        ev.append(f"Transaction occurred at night ({row.get('hour', '')}h).")
    
    amount = row['amount']
    ev.append(f"Amount: ${amount:.2f} via {row['payment_channel']} ({row['transaction_type']})")
    
    if len(ev) <= 2 and row['tslt_is_missing'] == 0:
        ev.append("No obvious behavioral anomalies detected in standard features.")
        
    return ev

def retrieve_similar_cases(row, retriever, corpus, preprocessor):
    NUM_HARD = ['amount','log_amount','spending_deviation_score','velocity_score',
               'geo_anomaly_score','hour','day_of_week','month','is_weekend','is_night',
               'tslt_abs','tslt_is_negative',
               'is_new_location_for_sender','is_new_payment_channel_for_sender',
               'is_new_transaction_type_for_sender','is_new_device_used_for_sender']
    CAT_COLS = ['transaction_type','merchant_category','payment_channel','device_used','location']
    df_row = pd.DataFrame([row])
    X = df_row[NUM_HARD + CAT_COLS]
    q_trans = preprocessor.transform(X)
    
    dists, inds = retriever.kneighbors(q_trans)
    sim_cases = []
    for d, idx in zip(dists[0], inds[0]):
        c_row = corpus.iloc[idx]
        sim_cases.append({
            'distance': float(d),
            'label': "Fraud" if c_row['is_fraud'] else "Legitimate",
            'amount': float(c_row['amount']),
            'channel': str(c_row['payment_channel']),
            'device': str(c_row['device_used']),
            'transaction_type': str(c_row['transaction_type'])
        })
    return sim_cases

def get_deterministic_fallback(risk_band):
    if risk_band in ["High", "Review", "Medium"]:
        title = "Why This Was Flagged"
        exp = "This transaction was selected for review based on its risk band and the listed structured evidence. Retrieved neighbors include historical labels for context only and are nearest-neighbor examples, not causal proof of fraud."
        if risk_band == "High":
            next_step = "Escalate to a senior analyst for manual review. If available, verify sender identity and compare the transaction with recent legitimate activity from the same sender. Do not block or freeze funds based only on this report."
        elif risk_band == "Review":
            next_step = "Perform manual review. If available, check whether the listed new attributes are consistent with recent legitimate activity from the same sender."
        else:
            next_step = "Monitor under current policy. If available, review recent sender activity before taking any additional action."
    else: # Low
        title = "Why This Was Not Escalated"
        exp = "This transaction was not escalated because it falls into the TSLT-missing structural bucket, which historically showed near-zero fraud in the training/validation data. This dependency may reflect a data artifact and should be monitored for drift."
        next_step = "No immediate review required under current policy. Monitor TSLT-missing rate and fraud rate over time."
    return title, exp, next_step

def get_caveats():
    return (
        "1. The underlying ML model is a weak ranking signal and high rank does not guarantee fraud.\n"
        "2. Similar historical cases are nearest-neighbor retrieval examples, not causal proof.\n"
        "3. The listed features are evidence for review routing, not definitive fraud indicators."
    )

def generate_guarded_brief_dict(transaction_id, score, risk_band, risk_factors, similar_cases, use_llm=True):
    title, exp, next_step = get_deterministic_fallback(risk_band)
    caveats = get_caveats()
    
    result = {
        "explanation_title": title,
        "explanation_text": exp,
        "next_step_title": "Suggested Analyst Next Step",
        "next_step_text": next_step,
        "caveats_title": "Caveats",
        "caveats_text": caveats,
        "status_msg": "LLM Status: Offline — deterministic fallback report is shown."
    }
    
    if not use_llm:
        return result
        
    is_online = check_lmstudio_status()
    if not is_online:
        return result

    client = OpenAI(base_url=get_lm_studio_url(), api_key="lm-studio")
    prompt = f"""You are a Fraud Analyst Co-pilot. Draft a candidate explanation for this transaction.
Risk Band: {risk_band}
Risk Score: {score}
Evidence: {', '.join(risk_factors)}
Provide a candidate explanation why this was flagged and a suggested next step."""
    
    try:
        response = client.chat.completions.create(
            model="qwen/qwen3-8b:2",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=300,
            timeout=5
        )
        llm_text = response.choices[0].message.content.strip()
        is_valid, _ = validate_guardrails(llm_text, risk_factors, risk_band, score)
        
        if is_valid:
            result["explanation_text"] = llm_text
            result["status_msg"] = "LLM Status: Online — Qwen3 candidate wording passed guardrail validation."
        else:
            result["status_msg"] = "LLM Status: Online — Qwen3 output failed guardrails; deterministic fallback used."
            
    except Exception as e:
        result["status_msg"] = "LLM Status: Offline — deterministic fallback report is shown."
        
    return result
