"""
FraudScorer Production Inference Pipeline API
=============================================
"""

import pickle
import pandas as pd
import numpy as np

class FraudScorer:
    def __init__(self, model_dir="models"):
        print(f"Initializing FraudScorer from {model_dir}...")
        with open(f"{model_dir}/preprocessor.pkl", "rb") as f:
            self.preprocessor = pickle.load(f)
        with open(f"{model_dir}/calibrated_lgbm.pkl", "rb") as f:
            self.model = pickle.load(f)
        with open(f"{model_dir}/retriever.pkl", "rb") as f:
            self.retriever = pickle.load(f)
            
        self.rag_corpus = pd.read_pickle(f"{model_dir}/rag_corpus.pkl")
        self.history = pd.read_pickle(f"{model_dir}/history_store.pkl")
        
        # Build quick lookups for memory efficiency
        self.seen_locations = set(zip(self.history['sender_account'], self.history['location']))
        self.seen_channels = set(zip(self.history['sender_account'], self.history['payment_channel']))
        self.seen_devices = set(zip(self.history['sender_account'], self.history['device_used']))
        self.seen_tx_types = set(zip(self.history['sender_account'], self.history['transaction_type']))
        
        self.NUM_HARD = ['amount','log_amount','spending_deviation_score','velocity_score',
                         'geo_anomaly_score','hour','day_of_week','month','is_weekend','is_night',
                         'tslt_abs','tslt_is_negative',
                         'is_new_location_for_sender','is_new_payment_channel_for_sender',
                         'is_new_transaction_type_for_sender','is_new_device_used_for_sender']
        self.CAT_COLS = ['transaction_type','merchant_category','payment_channel','device_used','location']
        
        print("FraudScorer Ready.")

    def _compute_realtime_features(self, tx_dict):
        # Convert to single row dataframe
        df = pd.DataFrame([tx_dict])
        
        # Ensure timestamp is datetime
        if 'timestamp' in df.columns and isinstance(df['timestamp'][0], str):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
        # Standard features
        df['log_amount'] = np.log1p(df['amount'])
        df['hour'] = df['timestamp'].dt.hour if 'timestamp' in df.columns else 12
        df['day_of_week'] = df['timestamp'].dt.dayofweek if 'timestamp' in df.columns else 0
        df['month'] = df['timestamp'].dt.month if 'timestamp' in df.columns else 1
        df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)
        df['is_night'] = ((df['hour']>=0)&(df['hour']<=5)).astype(int)
        
        tslt = df['time_since_last_transaction'].values[0]
        if pd.isna(tslt):
            df['tslt_abs'] = 0.0
            df['tslt_is_missing'] = 1
            df['tslt_is_negative'] = 0
        else:
            df['tslt_abs'] = abs(tslt)
            df['tslt_is_missing'] = 0
            df['tslt_is_negative'] = 1 if tslt < 0 else 0
            
        # Novelty Features via In-Memory Store
        sender = tx_dict.get('sender_account')
        df['is_new_location_for_sender'] = 0 if (sender, tx_dict.get('location')) in self.seen_locations else 1
        df['is_new_payment_channel_for_sender'] = 0 if (sender, tx_dict.get('payment_channel')) in self.seen_channels else 1
        df['is_new_device_used_for_sender'] = 0 if (sender, tx_dict.get('device_used')) in self.seen_devices else 1
        df['is_new_transaction_type_for_sender'] = 0 if (sender, tx_dict.get('transaction_type')) in self.seen_tx_types else 1
        
        return df

    def _generate_copilot_brief(self, df_row, score):
        # Determine Band
        if score >= 0.02: # Approx top 1%
            band = "High Risk"
            action = "Escalate to senior analyst"
        elif score >= 0.005: # Approx top 5%
            band = "Review Needed"
            action = "Manual review"
        else:
            return "No brief generated for Low Risk transactions."
            
        # RAG Retrieval
        trans_features = self.preprocessor.transform(df_row[self.NUM_HARD + self.CAT_COLS])
        dists, inds = self.retriever.kneighbors(trans_features)
        
        sim_cases = []
        for d, idx in zip(dists[0], inds[0]):
            c_row = self.rag_corpus.iloc[idx]
            sim_cases.append(f"- Dist: {d:.2f} | Label: {'Fraud' if c_row['is_fraud'] else 'Legit'} | Amount: ${c_row['amount']:.2f}")
            
        # Evidence
        ev = []
        if df_row['is_new_location_for_sender'].values[0] == 1:
            ev.append("- First time using this location.")
        if df_row['is_new_device_used_for_sender'].values[0] == 1:
            ev.append("- First time using this device.")
            
        # Rule 5: Explicitly say evidence is weak
        if len(ev) <= 2:
            ev.append("- Evidence is weak.")
            
        ev_str = "\n".join(ev)
        sim_str = "\n".join(sim_cases)
        
        # Rule 4 & 6: Why This Was Flagged
        if band == "High Risk":
            why = "This transaction shares risk indicators with known historical cases."
        else:
            why = "The model detected anomalies. The retrieved history shows mixed outcomes for similar patterns."
        
        brief = f"""
### AI Fraud Co-pilot Brief
**Risk Band**: {band} (Score: {score:.4f})
**Recommended Action**: {action}

**Detected Evidence**:
{ev_str}

**Historical Pattern Match**:
{sim_str}

**Why This Was Flagged**:
{why}

**Caveats**:
- *Weak Model Signal*: The underlying Cascade model is a weak ranking signal. High score does not guarantee fraud.
- *RAG Limitations*: Similar transactions retrieved are based on structured features, not explicit causal links.
- *Prototype System*: This brief is generated by a template-based prototype (LLM-ready); no real LLM call was used.
"""
        return brief

    def predict(self, transaction_dict):
        """
        Main entry point for Backend.
        Takes a JSON/Dict representing a single transaction.
        Returns a result Dict.
        """
        df = self._compute_realtime_features(transaction_dict)
        
        # Cascade Logic
        if df['tslt_is_missing'].values[0] == 1:
            score = 1e-6 # Hard rule for missing TSLT
        else:
            X_trans = self.preprocessor.transform(df[self.NUM_HARD + self.CAT_COLS])
            score = float(self.model.predict_proba(X_trans)[0, 1])
            
        result = {
            "transaction_id": transaction_dict.get("transaction_id", "UNKNOWN"),
            "fraud_probability": score,
            "status": "success",
            "copilot_brief": None
        }
        
        if score >= 0.005:
            result["copilot_brief"] = self._generate_copilot_brief(df, score)
            
        return result
