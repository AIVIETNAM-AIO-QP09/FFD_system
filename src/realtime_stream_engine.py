"""
Real-time Streaming Engine Module
=================================
Simulates a Stateful Stream Processing Engine (like Apache Flink / Kafka consumer).
Maintains per-account sliding window state in RAM, calculates real-time features (<20ms latency),
and performs Two-Tier Routing:
  - Tier 1: Auto-block if score > 0.80
  - Tier 2: Route to Review Queue (SQLite DB) if score between 0.02 and 0.80
"""

import os
import sys
import pickle
import time
import datetime
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(__file__))
from feedback_manager import add_to_review_queue

class RealtimeStreamEngine:
    def __init__(self, model_dir="models", auto_block_threshold=0.80, review_threshold=0.02):
        self.model_dir = model_dir
        self.auto_block_threshold = auto_block_threshold
        self.review_threshold = review_threshold
        
        # Load preprocessor and model
        with open(os.path.join(model_dir, "preprocessor.pkl"), "rb") as f:
            self.preprocessor = pickle.load(f)
        with open(os.path.join(model_dir, "calibrated_lgbm.pkl"), "rb") as f:
            self.model = pickle.load(f)
            
        # In-memory sliding window state store per sender_account
        # Key: sender_account, Value: list of dicts [{'timestamp': dt, 'amount': float, 'location': str, ...}]
        self.state_store = {}
        
        # Load history lookup tables for novelty detection if available
        hist_path = os.path.join(model_dir, "history_store.pkl")
        if os.path.exists(hist_path):
            hist = pd.read_pickle(hist_path)
            self.seen_locations = set(zip(hist['sender_account'], hist['location']))
            self.seen_channels = set(zip(hist['sender_account'], hist['payment_channel']))
            self.seen_devices = set(zip(hist['sender_account'], hist['device_used']))
            self.seen_tx_types = set(zip(hist['sender_account'], hist['transaction_type']))
        else:
            self.seen_locations = set()
            self.seen_channels = set()
            self.seen_devices = set()
            self.seen_tx_types = set()
            
        self.NUM_HARD = ['amount','log_amount','spending_deviation_score','velocity_score',
                         'geo_anomaly_score','hour','day_of_week','month','is_weekend','is_night',
                         'tslt_abs','tslt_is_negative',
                         'is_new_location_for_sender','is_new_payment_channel_for_sender',
                         'is_new_transaction_type_for_sender','is_new_device_used_for_sender']
        self.CAT_COLS = ['transaction_type','merchant_category','payment_channel','device_used','location']
        
        # Metrics tracking
        self.stats = {
            "PROCESSED": 0,
            "AUTO_BLOCKED": 0,
            "REVIEW_QUEUED": 0,
            "PASSED": 0,
            "AVG_LATENCY_MS": 0.0
        }
        self._total_time_ms = 0.0

    def process_event(self, tx_dict):
        """
        Processes a single transaction event in real-time.
        Returns: (status, risk_score, risk_band, latency_ms)
        status: 'AUTO_BLOCKED', 'REVIEW_QUEUED', or 'PASSED'
        """
        t0 = time.perf_counter()
        self.stats["PROCESSED"] += 1
        
        sender = str(tx_dict.get('sender_account', 'UNKNOWN'))
        ts_val = tx_dict.get('timestamp')
        if isinstance(ts_val, str):
            try:
                tx_dt = pd.to_datetime(ts_val)
            except Exception:
                tx_dt = datetime.datetime.now()
        elif isinstance(ts_val, (datetime.datetime, pd.Timestamp)):
            tx_dt = ts_val
        else:
            tx_dt = datetime.datetime.now()
            
        amount = float(tx_dict.get('amount', 0.0))
        
        # 1. Compute stateful sliding window features (< 5ms)
        account_history = self.state_store.get(sender, [])
        
        if len(account_history) > 0:
            last_dt = account_history[-1]['timestamp']
            tslt_seconds = (tx_dt - last_dt).total_seconds()
        else:
            # Check if raw dict had time_since_last_transaction
            raw_tslt = tx_dict.get('time_since_last_transaction')
            tslt_seconds = float(raw_tslt) if raw_tslt is not None and not pd.isna(raw_tslt) else None
            
        # Update sliding window state store (keep last 50 transactions per account)
        account_history.append({'timestamp': tx_dt, 'amount': amount})
        if len(account_history) > 50:
            account_history = account_history[-50:]
        self.state_store[sender] = account_history
        
        # 2. Build feature vector for LightGBM
        row = dict(tx_dict)
        row['amount'] = amount
        row['log_amount'] = np.log1p(amount)
        row['hour'] = tx_dt.hour
        row['day_of_week'] = tx_dt.dayofweek
        row['month'] = tx_dt.month
        row['is_weekend'] = 1 if tx_dt.dayofweek in [5, 6] else 0
        row['is_night'] = 1 if 0 <= tx_dt.hour <= 5 else 0
        
        if tslt_seconds is None or pd.isna(tslt_seconds):
            row['tslt_abs'] = 0.0
            row['tslt_is_negative'] = 0
            row['time_since_last_transaction'] = None
        else:
            row['tslt_abs'] = abs(tslt_seconds)
            row['tslt_is_negative'] = 1 if tslt_seconds < 0 else 0
            row['time_since_last_transaction'] = tslt_seconds
            
        # Fallback scores if missing
        row['spending_deviation_score'] = float(tx_dict.get('spending_deviation_score', 0.0))
        row['velocity_score'] = float(tx_dict.get('velocity_score', 0.0))
        row['geo_anomaly_score'] = float(tx_dict.get('geo_anomaly_score', 0.0))
        
        # Novelty features against historical lookup
        row['is_new_location_for_sender'] = 0 if (sender, tx_dict.get('location')) in self.seen_locations else 1
        row['is_new_payment_channel_for_sender'] = 0 if (sender, tx_dict.get('payment_channel')) in self.seen_channels else 1
        row['is_new_transaction_type_for_sender'] = 0 if (sender, tx_dict.get('transaction_type')) in self.seen_tx_types else 1
        row['is_new_device_used_for_sender'] = 0 if (sender, tx_dict.get('device_used')) in self.seen_devices else 1
        
        # Update lookup sets
        self.seen_locations.add((sender, tx_dict.get('location')))
        self.seen_channels.add((sender, tx_dict.get('payment_channel')))
        self.seen_tx_types.add((sender, tx_dict.get('transaction_type')))
        self.seen_devices.add((sender, tx_dict.get('device_used')))
        
        # 3. Model Scoring
        df_feature = pd.DataFrame([row])
        X_transformed = self.preprocessor.transform(df_feature[self.NUM_HARD + self.CAT_COLS])
        score = float(self.model.predict_proba(X_transformed)[:, 1][0])
        
        # Determine risk band
        if score >= 0.02:
            risk_band = "High"
        elif score >= 0.005:
            risk_band = "Review"
        else:
            risk_band = "Low"
            
        # 4. Two-Tier Routing Decision
        if score >= self.auto_block_threshold:
            status = "AUTO_BLOCKED"
            self.stats["AUTO_BLOCKED"] += 1
        elif score >= self.review_threshold:
            status = "REVIEW_QUEUED"
            self.stats["REVIEW_QUEUED"] += 1
            # Push to SQLite Review Queue
            row['timestamp'] = tx_dt.isoformat()
            add_to_review_queue(row, score, risk_band)
        else:
            status = "PASSED"
            self.stats["PASSED"] += 1
            
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0
        self._total_time_ms += latency_ms
        self.stats["AVG_LATENCY_MS"] = self._total_time_ms / self.stats["PROCESSED"]
        
        return status, score, risk_band, latency_ms

    def get_engine_stats(self):
        return self.stats
