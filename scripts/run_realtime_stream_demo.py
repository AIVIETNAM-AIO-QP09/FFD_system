"""
Real-time Streaming Pipeline Demo Runner
========================================
Simulates high-speed live transaction ingestion from test set into the Real-time Streaming Engine.
Demonstrates sub-20ms windowed feature calculation, Tier-1 Auto-blocking, and Tier-2 Review Queue push.
"""

import os
import sys
import argparse
import time
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from realtime_stream_engine import RealtimeStreamEngine
from feedback_manager import get_queue_stats, init_db

def run_realtime_simulation(num_events=200, delay_ms=5):
    print("="*70)
    print("🚀 STARTING REAL-TIME STREAMING FFD PIPELINE SIMULATION")
    print("="*70)
    
    init_db()
    
    test_csv = os.path.join("data", "split", "test.csv")
    if not os.path.exists(test_csv):
        print(f"❌ Error: {test_csv} not found. Please run python split_data.py first.")
        return
        
    print(f"📥 Loading stream source from {test_csv}...")
    df = pd.read_csv(test_csv, nrows=num_events * 2)
    
    # Sort or shuffle lightly to simulate real-time stream
    df = df.sample(n=min(num_events, len(df)), random_state=42).reset_index(drop=True)
    
    print(f"⚡ Initializing RealtimeStreamEngine (Stateful RAM Window + LightGBM Preprocessor)...")
    engine = RealtimeStreamEngine(model_dir="models", auto_block_threshold=0.80, review_threshold=0.02)
    
    print("\n--- STREAMING EVENTS ---")
    start_time = time.time()
    
    for idx, row in df.iterrows():
        tx_dict = row.to_dict()
        
        status, score, band, latency = engine.process_event(tx_dict)
        
        tx_id = tx_dict.get('transaction_id', f'TX_{idx}')
        amt = tx_dict.get('amount', 0.0)
        
        if status == "AUTO_BLOCKED":
            symbol = "🛑 [AUTO-BLOCK]"
        elif status == "REVIEW_QUEUED":
            symbol = "📋 [REVIEW QUEUE]"
        else:
            symbol = "✅ [PASS]"
            
        # Print status line
        print(f"[{idx+1:03d}/{len(df):03d}] {symbol:<18} ID: {tx_id:<12} | Amt: ${amt:8.2f} | Score: {score:.4f} ({band:6}) | Latency: {latency:.2f}ms")
        
        if delay_ms > 0:
            time.sleep(delay_ms / 1000.0)
            
    total_duration = time.time() - start_time
    stats = engine.get_engine_stats()
    q_stats = get_queue_stats()
    
    print("\n" + "="*70)
    print("📊 REAL-TIME STREAMING SIMULATION SUMMARY")
    print("="*70)
    print(f"⏱️ Total Simulation Time : {total_duration:.2f} seconds")
    print(f"⚡ Throughput            : {stats['PROCESSED'] / total_duration:.1f} tx/sec")
    print(f"🚀 Average Latency       : {stats['AVG_LATENCY_MS']:.2f} ms per transaction")
    print("-" * 50)
    print(f"✅ Passed Tier-1 (<0.02) : {stats['PASSED']}")
    print(f"📋 Queued Tier-2 (0.02-0.8): {stats['REVIEW_QUEUED']} -> Pushed to SQLite Review Queue")
    print(f"🛑 Auto-Blocked  (>=0.8) : {stats['AUTO_BLOCKED']}")
    print("-" * 50)
    print(f"🗄️ SQLite Queue Status   : {q_stats}")
    print("="*70)
    print("👉 Next step: Launch Streamlit Dashboard (`streamlit run app_streamlit.py`) to review the live queue!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Real-time Streaming FFD Simulation")
    parser.add_argument("--num_events", type=int, default=100, help="Number of streaming events to process")
    parser.add_argument("--delay_ms", type=int, default=5, help="Delay between events in milliseconds")
    args = parser.parse_args()
    
    run_realtime_simulation(num_events=args.num_events, delay_ms=args.delay_ms)
