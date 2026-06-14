from pathlib import Path
import time

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
TARGET_COL = "is_fraud"

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data/processed/cleaned_baseline_data.csv"
SPLIT_DIR = PROJECT_ROOT / "data/splits"
MODEL_DIR = PROJECT_ROOT / "models"
EXPERIMENT_DIR = PROJECT_ROOT / "experiments"


# Chỉ dùng các cột cần thiết, không dùng toàn bộ dataset.
# Không dùng: is_fraud, fraud_type, timestamp raw, transaction_id, sender_account,
# receiver_account, ip_address, device_hash vì có thể là target/leakage/ID.
NUMERICAL_COLS = [
    "amount_log",
    "time_since_last_transaction",
    "spending_deviation_score",
    "velocity_score",
    "geo_anomaly_score",
    "transaction_hour",
    "transaction_day_of_week",
    "is_weekend",
]

CATEGORICAL_COLS = [
    "transaction_type",
    "merchant_category",
    "device_used",
    "payment_channel",
    "location",
]


def load_indices(path):
    """
    Đọc file index đã được tạo từ create_splits.py.

    Ví dụ:
    data/splits/train_indices.csv
    data/splits/dev_indices.csv
    data/splits/test_indices.csv
    """
    indices_df = pd.read_csv(path)
    return indices_df["row_index"].tolist()


def main():
    start_time = time.time()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading processed data...")
    df = pd.read_csv(DATA_PATH)

    print("Loaded data shape:", df.shape)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COL}")

    print("Loading fixed split indices...")
    train_indices = load_indices(SPLIT_DIR / "train_indices.csv")
    dev_indices = load_indices(SPLIT_DIR / "dev_indices.csv")
    test_indices = load_indices(SPLIT_DIR / "test_indices.csv")

    train_df = df.loc[train_indices]
    dev_df = df.loc[dev_indices]
    test_df = df.loc[test_indices]

    # Chỉ giữ lại những cột thật sự tồn tại trong dataframe.
    numerical_cols = [col for col in NUMERICAL_COLS if col in df.columns]
    categorical_cols = [col for col in CATEGORICAL_COLS if col in df.columns]

    feature_cols = numerical_cols + categorical_cols

    print("Numerical columns used:", numerical_cols)
    print("Categorical columns used:", categorical_cols)
    print("All feature columns used:", feature_cols)

    if len(feature_cols) == 0:
        raise ValueError("No feature columns found. Please check column names.")

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COL].astype(int)

    X_dev = dev_df[feature_cols]
    y_dev = dev_df[TARGET_COL].astype(int)

    X_test = test_df[feature_cols]
    y_test = test_df[TARGET_COL].astype(int)

    print("Train shape:", X_train.shape)
    print("Dev shape:", X_dev.shape)
    print("Test shape:", X_test.shape)

    print("Train fraud rate:", y_train.mean())
    print("Dev fraud rate:", y_dev.mean())
    print("Test fraud rate:", y_test.mean())

    print("Missing values in selected features:")
    print(df[feature_cols].isna().sum())

    # Pipeline xử lý cột số:
    # 1. SimpleImputer(strategy="median"):
    #    Nếu cột số có NaN thì điền bằng median.
    # 2. StandardScaler:
    #    Chuẩn hóa cột số để Logistic Regression học ổn định hơn.
    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Pipeline xử lý cột chữ:
    # 1. SimpleImputer(strategy="constant", fill_value="Unknown"):
    #    Nếu cột chữ có NaN thì điền bằng "Unknown".
    # 2. OneHotEncoder:
    #    Biến dữ liệu chữ thành dạng số 0/1 để model hiểu được.
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                    min_frequency=50,
                ),
            ),
        ]
    )

    # ColumnTransformer giúp áp dụng cách xử lý khác nhau cho từng nhóm cột.
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, numerical_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )

    # Baseline model: Logistic Regression.
    # class_weight="balanced" giúp model chú ý hơn tới class fraud,
    # vì fraud thường ít hơn non-fraud rất nhiều.
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="saga",
        random_state=RANDOM_STATE,
    )

    # Pipeline tổng:
    # Bước 1: xử lý dữ liệu
    # Bước 2: train model
    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    print("Training baseline Logistic Regression model...")
    clf.fit(X_train, y_train)

    print("Evaluating on Dev set...")
    y_dev_pred = clf.predict(X_dev)
    y_dev_proba = clf.predict_proba(X_dev)[:, 1]

    accuracy = accuracy_score(y_dev, y_dev_pred)
    precision = precision_score(y_dev, y_dev_pred, zero_division=0)
    recall = recall_score(y_dev, y_dev_pred, zero_division=0)
    f1 = f1_score(y_dev, y_dev_pred, zero_division=0)
    roc_auc = roc_auc_score(y_dev, y_dev_proba)
    pr_auc = average_precision_score(y_dev, y_dev_proba)
    cm = confusion_matrix(y_dev, y_dev_pred)

    runtime_seconds = round(time.time() - start_time, 2)

    model_path = MODEL_DIR / "baseline_model.pkl"
    metrics_path = EXPERIMENT_DIR / "baseline_metrics.txt"

    # Lưu cả preprocessing + model vào file .pkl.
    # Sau này load lại file này là predict được luôn.
    joblib.dump(clf, model_path)

    report = f"""
BASELINE MODEL REPORT - WEEK 1

Branch:
feature/model-baseline-week1-streamlit

Split:
Train / Dev / Test = 80% / 15% / 5%

Model:
Logistic Regression

Preprocessing:
Numerical features:
{numerical_cols}

Categorical features:
{categorical_cols}

Numerical preprocessing:
SimpleImputer(strategy="median") + StandardScaler

Categorical preprocessing:
SimpleImputer(strategy="constant", fill_value="Unknown") + OneHotEncoder(handle_unknown="ignore", sparse_output=True, min_frequency=50)

Data size:
Train size: {len(X_train)}
Dev size: {len(X_dev)}
Test size: {len(X_test)}

Fraud rate:
Train fraud rate: {y_train.mean()}
Dev fraud rate: {y_dev.mean()}
Test fraud rate: {y_test.mean()}

Dev metrics:
Accuracy: {accuracy}
Precision: {precision}
Recall: {recall}
F1-score: {f1}
ROC-AUC: {roc_auc}
PR-AUC: {pr_auc}

Confusion matrix:
{cm}

Classification report:
{classification_report(y_dev, y_dev_pred, zero_division=0)}

Runtime seconds:
{runtime_seconds}

Saved model:
{model_path}
"""

    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(report)
    print("Saved model to:", model_path)
    print("Saved metrics to:", metrics_path)


if __name__ == "__main__":
    main()