import pandas as pd
from pathlib import Path
import os
import sys

# Force UTF-8 encoding for stdout if possible, or just use English text
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None

def main():
    raw_data_path = "data/raw/financial_fraud_detection_dataset.csv"
    output_dir = "data/split"
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("1. Reading raw data (this might take a few minutes for 5M rows)...")
    df = pd.read_csv(raw_data_path)
    
    print("2. Converting timestamp column to Datetime...")
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
    
    print("3. Sorting data by timestamp (Out-Of-Time Split)...")
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    split_index = int(len(df) * 0.8)
    cutoff_date = df.loc[split_index, 'timestamp']
    print(f"--> Cut-off Date: {cutoff_date}")
    
    print("4. Splitting Train and Test sets...")
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]
    
    print(f"--> Train set size: {len(train_df)}")
    print(f"--> Test set size: {len(test_df)}")
    
    print("5. Saving Train set to CSV...")
    train_path = os.path.join(output_dir, "train.csv")
    train_df.to_csv(train_path, index=False)
    print(f"--> Successfully saved: {train_path}")
    
    print("6. Saving Test set to CSV...")
    test_path = os.path.join(output_dir, "test.csv")
    test_df.to_csv(test_path, index=False)
    print(f"--> Successfully saved: {test_path}")
    
    print("\n--- DONE ---")

if __name__ == "__main__":
    main()
