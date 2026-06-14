from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
TARGET_COL = "is_fraud"

# Test nhanh trước với 50,000 dòng.
# Khi pipeline chạy ổn, có thể đổi thành 100000 hoặc None để chạy full data.
SAMPLE_SIZE = 50000

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data/processed/cleaned_baseline_data.csv"
SPLIT_DIR = PROJECT_ROOT / "data/splits"


def main():
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading processed data...")
    df = pd.read_csv(DATA_PATH)

    print("Original data shape:", df.shape)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COL}")

    if SAMPLE_SIZE is not None and len(df) > SAMPLE_SIZE:
        df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)

    print("Data used for split:", df.shape)

    print("Target distribution:")
    print(df[TARGET_COL].value_counts(normalize=True))

    y = df[TARGET_COL]

    # Chia lần 1:
    # Train = 80%
    # Temp = 20%
    train_df, temp_df = train_test_split(
        df,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Chia lần 2:
    # Temp đang là 20%.
    # Trong 20% đó, Dev cần 15%, Test cần 5%.
    # Test chiếm 5 / 20 = 0.25 của temp.
    y_temp = temp_df[TARGET_COL]

    dev_df, test_df = train_test_split(
        temp_df,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    total = len(df)

    print("Train shape:", train_df.shape)
    print("Dev shape:", dev_df.shape)
    print("Test shape:", test_df.shape)

    print("Train ratio:", len(train_df) / total)
    print("Dev ratio:", len(dev_df) / total)
    print("Test ratio:", len(test_df) / total)

    pd.DataFrame({"row_index": train_df.index}).to_csv(
        SPLIT_DIR / "train_indices.csv",
        index=False,
    )

    pd.DataFrame({"row_index": dev_df.index}).to_csv(
        SPLIT_DIR / "dev_indices.csv",
        index=False,
    )

    pd.DataFrame({"row_index": test_df.index}).to_csv(
        SPLIT_DIR / "test_indices.csv",
        index=False,
    )

    split_summary = pd.DataFrame(
        {
            "split": ["train", "dev", "test"],
            "num_rows": [len(train_df), len(dev_df), len(test_df)],
            "ratio": [
                len(train_df) / total,
                len(dev_df) / total,
                len(test_df) / total,
            ],
            "fraud_rate": [
                train_df[TARGET_COL].mean(),
                dev_df[TARGET_COL].mean(),
                test_df[TARGET_COL].mean(),
            ],
        }
    )

    split_summary.to_csv(SPLIT_DIR / "split_summary.csv", index=False)

    print("Saved split files to:", SPLIT_DIR)
    print(split_summary)


if __name__ == "__main__":
    main()