"""
Feedback Manager Module
=======================
Manages the Tier-2 Review Queue and Human-in-the-Loop Analyst Feedback Database using SQLite.
Enables Active Learning by capturing human domain decisions to retrain models.
"""

import sqlite3
import json
import os
import datetime
import pandas as pd

DB_PATH = os.path.join("data", "ffd_database.sqlite")

def get_db_connection(db_path=DB_PATH):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=10.0)
    conn.row_factory = sqlite3.Row
    return conn

def init_db(db_path=DB_PATH):
    """Initializes SQLite tables for Review Queue and Analyst Feedback."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    
    # Table 1: Review Queue (Transactions flagged by Tier-1 Real-time Engine for human review)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS review_queue (
        transaction_id TEXT PRIMARY KEY,
        timestamp TEXT,
        amount REAL,
        sender_account TEXT,
        receiver_account TEXT,
        transaction_type TEXT,
        payment_channel TEXT,
        device_used TEXT,
        location TEXT,
        merchant_category TEXT,
        time_since_last_transaction REAL,
        risk_score REAL,
        risk_band TEXT,
        status TEXT DEFAULT 'PENDING',
        raw_json TEXT
    )
    """)
    
    # Table 2: Analyst Feedback (Closed-loop ground truth labels from analysts)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS analyst_feedback (
        feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
        transaction_id TEXT,
        analyst_decision TEXT,
        is_confirmed_fraud INTEGER,
        analyst_notes TEXT,
        reviewed_at TEXT,
        raw_json TEXT
    )
    """)
    
    conn.commit()
    conn.close()

def add_to_review_queue(tx_dict, risk_score, risk_band, db_path=DB_PATH):
    """Adds a flagged transaction from the real-time engine into the review queue."""
    init_db(db_path)
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    
    tx_id = str(tx_dict.get('transaction_id', 'UNKNOWN'))
    ts = str(tx_dict.get('timestamp', datetime.datetime.now().isoformat()))
    amount = float(tx_dict.get('amount', 0.0))
    sender = str(tx_dict.get('sender_account', ''))
    receiver = str(tx_dict.get('receiver_account', ''))
    tx_type = str(tx_dict.get('transaction_type', ''))
    channel = str(tx_dict.get('payment_channel', ''))
    device = str(tx_dict.get('device_used', ''))
    location = str(tx_dict.get('location', ''))
    merchant = str(tx_dict.get('merchant_category', ''))
    tslt = tx_dict.get('time_since_last_transaction')
    tslt_val = float(tslt) if tslt is not None and not pd.isna(tslt) else None
    
    raw_json = json.dumps(tx_dict, default=str)
    
    cursor.execute("""
    INSERT OR REPLACE INTO review_queue (
        transaction_id, timestamp, amount, sender_account, receiver_account,
        transaction_type, payment_channel, device_used, location, merchant_category,
        time_since_last_transaction, risk_score, risk_band, status, raw_json
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'PENDING', ?)
    """, (tx_id, ts, amount, sender, receiver, tx_type, channel, device, location, merchant, tslt_val, float(risk_score), risk_band, raw_json))
    
    conn.commit()
    conn.close()

def get_pending_queue(limit=100, db_path=DB_PATH):
    """Retrieves pending transactions awaiting analyst review, ordered by risk score descending."""
    init_db(db_path)
    conn = get_db_connection(db_path)
    df = pd.read_sql_query("""
        SELECT * FROM review_queue 
        WHERE status = 'PENDING' 
        ORDER BY risk_score DESC 
        LIMIT ?
    """, conn, params=(limit,))
    conn.close()
    if not df.empty and 'raw_json' in df.columns:
        parsed_rows = []
        for _, r in df.iterrows():
            try:
                data = json.loads(r['raw_json'])
                data['risk_score'] = r['risk_score']
                data['risk_band'] = r['risk_band']
                parsed_rows.append(data)
            except Exception:
                parsed_rows.append(r.to_dict())
        return pd.DataFrame(parsed_rows)
    return df

def get_transaction_from_queue(transaction_id, db_path=DB_PATH):
    """Fetches a specific transaction row from the queue."""
    init_db(db_path)
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT raw_json, risk_score, risk_band FROM review_queue WHERE transaction_id = ?", (str(transaction_id),))
    row = cursor.fetchone()
    conn.close()
    if row:
        tx_data = json.loads(row['raw_json'])
        return tx_data, row['risk_score'], row['risk_band']
    return None, None, None

def submit_analyst_feedback(transaction_id, decision, is_confirmed_fraud, notes, raw_json_str=None, db_path=DB_PATH):
    """
    Records human analyst decision and updates queue status.
    decision: 'CONFIRM_FRAUD', 'FALSE_ALARM', 'ESCALATE'
    is_confirmed_fraud: 1 (Fraud), 0 (Legit)
    """
    init_db(db_path)
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    
    # If raw_json_str not provided, fetch from review_queue
    if not raw_json_str:
        cursor.execute("SELECT raw_json FROM review_queue WHERE transaction_id = ?", (str(transaction_id),))
        r = cursor.fetchone()
        raw_json_str = r['raw_json'] if r else "{}"
        
    now_str = datetime.datetime.now().isoformat()
    
    cursor.execute("""
    INSERT INTO analyst_feedback (
        transaction_id, analyst_decision, is_confirmed_fraud, analyst_notes, reviewed_at, raw_json
    ) VALUES (?, ?, ?, ?, ?, ?)
    """, (str(transaction_id), decision, int(is_confirmed_fraud), notes, now_str, raw_json_str))
    
    cursor.execute("""
    UPDATE review_queue SET status = ? WHERE transaction_id = ?
    """, (f"REVIEWED_{decision}", str(transaction_id)))
    
    conn.commit()
    conn.close()

def get_all_feedback(db_path=DB_PATH):
    """Retrieves all labeled human feedback for active learning retraining."""
    init_db(db_path)
    conn = get_db_connection(db_path)
    df = pd.read_sql_query("SELECT * FROM analyst_feedback ORDER BY reviewed_at DESC", conn)
    conn.close()
    return df

def get_queue_stats(db_path=DB_PATH):
    """Returns summary counts for dashboard status panel."""
    init_db(db_path)
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT status, COUNT(*) as cnt FROM review_queue GROUP BY status")
    rows = cursor.fetchall()
    stats = {row['status']: row['cnt'] for row in rows}
    
    cursor.execute("SELECT COUNT(*) as cnt FROM analyst_feedback")
    fb_cnt = cursor.fetchone()['cnt']
    stats['TOTAL_FEEDBACK'] = fb_cnt
    conn.close()
    return stats
