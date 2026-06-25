import json
import sys
sys.path.append('src')
from inference import FraudScorer

def main():
    scorer = FraudScorer(model_dir="models")
    
    # Simulate a High-Risk Transaction (TSLT not missing, High Amount, New Device)
    tx_high = {
        "transaction_id": "TXN_LIVE_001",
        "timestamp": "2024-10-15T03:30:00Z",
        "sender_account": "ACC_DEMO_999",
        "receiver_account": "ACC_RCV_888",
        "amount": 9500.0,
        "transaction_type": "transfer",
        "merchant_category": "crypto",
        "location": "Moscow",
        "device_used": "desktop",
        "payment_channel": "card",
        "time_since_last_transaction": 45.0,
        "spending_deviation_score": 5.2,
        "velocity_score": 25,
        "geo_anomaly_score": 0.95
    }
    
    # Simulate a Safe Transaction (TSLT missing)
    tx_safe = {
        "transaction_id": "TXN_LIVE_002",
        "timestamp": "2024-10-15T14:30:00Z",
        "sender_account": "ACC_DEMO_111",
        "receiver_account": "ACC_RCV_222",
        "amount": 15.0,
        "transaction_type": "payment",
        "merchant_category": "retail",
        "location": "New York",
        "device_used": "mobile",
        "payment_channel": "app",
        "time_since_last_transaction": None,
        "spending_deviation_score": 0.1,
        "velocity_score": 2,
        "geo_anomaly_score": 0.05
    }
    
    print("\n--- Scoring TXN_LIVE_001 (High Risk) ---")
    res_high = scorer.predict(tx_high)
    print(json.dumps(res_high, indent=2))
    
    if res_high.get("copilot_brief"):
        print("\n> Generated Co-pilot Brief:")
        print(res_high["copilot_brief"])
        
    print("\n--- Scoring TXN_LIVE_002 (Safe) ---")
    res_safe = scorer.predict(tx_safe)
    print(json.dumps(res_safe, indent=2))

if __name__ == "__main__":
    main()
