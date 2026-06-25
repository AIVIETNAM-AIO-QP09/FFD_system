"""
Sanity Check Script
===================
1. Label Permutation Test: So sánh PR-AUC với label thật vs label shuffle
2. Single-feature PR-AUC diagnostics cho tất cả feature hiện tại
3. Fraud rate uniformity test: xác minh categorical có flat fraud rate không

Chỉ dùng train.csv. Không mở test.csv.
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

REPORT_PATH = "reports/sanity_check_report.md"
TRAIN_PATH = "data/split/train.csv"

def write_md(content, mode='a'):
    with open(REPORT_PATH, mode, encoding='utf-8') as f:
        f.write(content + "\n")

def get_top_k_prec(probs, labels, pct):
    k = max(1, int(len(probs) * pct))
    idx = np.argsort(probs)[::-1][:k]
    return labels.iloc[idx].mean() if hasattr(labels, 'iloc') else labels[idx].mean()

def load_and_prepare():
    print("Loading data...")
    df = pd.read_csv(TRAIN_PATH, parse_dates=['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    df['log_amount'] = np.log1p(df['amount'])
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_night'] = ((df['hour'] >= 0) & (df['hour'] <= 5)).astype(int)
    df['tslt_is_missing'] = df['time_since_last_transaction'].isnull().astype(int)
    df['tslt_is_negative'] = (df['time_since_last_transaction'] < 0).astype(int)
    df['tslt_abs'] = df['time_since_last_transaction'].abs()

    train = df.iloc[:3200000].copy()
    val = df.iloc[3200000:4000000].copy()
    return train, val

def single_feature_pr_auc(col_values, labels, direction='positive'):
    """Test a single numeric feature. Tries both directions."""
    try:
        pr_pos = average_precision_score(labels, col_values)
        pr_neg = average_precision_score(labels, -col_values)
        return max(pr_pos, pr_neg), ('pos' if pr_pos >= pr_neg else 'neg')
    except Exception:
        return None, None

def run_single_feature_diagnostics(train, val):
    print("Running single-feature diagnostics...")
    y_val = val['is_fraud'].astype(int)
    val_fraud_rate = y_val.mean()

    num_features = [
        'amount', 'log_amount', 'spending_deviation_score', 'velocity_score',
        'geo_anomaly_score', 'hour', 'day_of_week', 'month',
        'is_weekend', 'is_night', 'tslt_abs', 'tslt_is_missing', 'tslt_is_negative'
    ]

    content = "## G. New Feature Signal Diagnostics (Single-Feature PR-AUC)\n"
    content += f"Validation fraud rate (random baseline): **{val_fraud_rate:.5f}**\n\n"
    content += "| Feature | PR-AUC | Rel Lift | Direction | Assessment |\n"
    content += "|---|---|---|---|---|\n"

    for feat in num_features:
        if feat not in val.columns:
            continue
        col = val[feat].fillna(val[feat].median() if not val[feat].isna().all() else 0)
        pr, direction = single_feature_pr_auc(col, y_val)
        if pr is None:
            continue
        rel_lift = pr / val_fraud_rate
        assess = "✅ Signal" if rel_lift > 1.1 else ("⚠️ Marginal" if rel_lift > 1.02 else "❌ Noise")
        content += f"| {feat} | {pr:.5f} | {rel_lift:.3f}x | {direction} | {assess} |\n"

    # Categorical features via OHE encoding
    cat_features = ['transaction_type', 'merchant_category', 'payment_channel', 'device_used', 'location']
    for feat in cat_features:
        # Use fraud rate per category as score
        rates = train.groupby(feat)['is_fraud'].mean().to_dict()
        col = val[feat].map(rates).fillna(val_fraud_rate)
        pr, direction = single_feature_pr_auc(col, y_val)
        if pr is None:
            continue
        rel_lift = pr / val_fraud_rate
        assess = "✅ Signal" if rel_lift > 1.1 else ("⚠️ Marginal" if rel_lift > 1.02 else "❌ Noise")
        content += f"| {feat} (mean-encoded) | {pr:.5f} | {rel_lift:.3f}x | {direction} | {assess} |\n"

    write_md(content)

def run_label_permutation_test(train, val):
    print("Running label permutation test...")
    y_train_real = train['is_fraud'].astype(int)
    y_train_shuffled = y_train_real.sample(frac=1, random_state=42).reset_index(drop=True)
    y_val = val['is_fraud'].astype(int)
    val_fraud_rate = y_val.mean()

    num_cols = ['amount', 'log_amount', 'spending_deviation_score', 'velocity_score',
                'geo_anomaly_score', 'hour', 'day_of_week', 'month',
                'is_weekend', 'is_night', 'tslt_abs', 'tslt_is_missing', 'tslt_is_negative']
    cat_cols = ['transaction_type', 'merchant_category', 'payment_channel', 'device_used', 'location']
    all_cols = num_cols + cat_cols

    X_train = train[all_cols]
    X_val = val[all_cols]

    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median'))])
    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                                               ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ])

    results = []
    for label_name, y_train in [('Real Labels', y_train_real), ('Shuffled Labels', y_train_shuffled)]:
        imbalance = (len(y_train) - y_train.sum()) / y_train.sum()
        model = LGBMClassifier(scale_pos_weight=imbalance, n_estimators=100,
                               random_state=42, n_jobs=-1)
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
        print(f"  Training with {label_name}...")
        pipeline.fit(X_train, y_train)
        probs = pipeline.predict_proba(X_val)[:, 1]
        pr = average_precision_score(y_val, probs)
        roc = roc_auc_score(y_val, probs)
        p1 = get_top_k_prec(probs, y_val, 0.01)
        rel_lift = pr / val_fraud_rate
        results.append({
            'Label': label_name, 'PR-AUC': pr, 'ROC-AUC': roc,
            'Prec@1%': p1, 'Rel Lift': rel_lift
        })

    content = "## K. Label Permutation Sanity Check\n"
    content += "| Label Type | PR-AUC | ROC-AUC | Prec@1% | Rel Lift |\n"
    content += "|---|---|---|---|---|\n"
    for r in results:
        content += f"| {r['Label']} | {r['PR-AUC']:.5f} | {r['ROC-AUC']:.5f} | {r['Prec@1%']*100:.2f}% | {r['Rel Lift']:.3f}x |\n"

    pr_real = results[0]['PR-AUC']
    pr_shuffle = results[1]['PR-AUC']
    gap = pr_real - pr_shuffle
    gap_pct = (gap / pr_shuffle) * 100 if pr_shuffle > 0 else 0

    if gap_pct < 5:
        verdict = "🔴 **CONFIRMED: Dataset has near-zero predictive signal.** Real labels perform virtually identically to shuffled labels. The model is learning from noise, not from genuine fraud patterns."
    elif gap_pct < 20:
        verdict = "🟡 **WEAK SIGNAL**: Real labels marginally outperform shuffle. Some signal exists but it is very weak."
    else:
        verdict = "🟢 **SIGNAL EXISTS**: Real labels significantly outperform shuffle. Dataset contains learnable patterns."

    content += f"\n**Gap (Real vs Shuffle)**: {gap:+.5f} ({gap_pct:+.1f}%)\n\n**Verdict**: {verdict}\n"
    write_md(content)

    return pr_real, pr_shuffle, gap_pct

def run_fraud_rate_uniformity_check(train):
    print("Running fraud rate uniformity check...")
    val_fraud_rate = train['is_fraud'].mean()
    content = "## B. Current Evidence Summary — Fraud Rate Uniformity\n"
    content += f"**Overall fraud rate**: {val_fraud_rate:.5f}\n\n"

    cats = ['transaction_type', 'merchant_category', 'payment_channel', 'device_used', 'location']
    for cat in cats:
        agg = train.groupby(cat)['is_fraud'].agg(['count', 'sum', 'mean']).reset_index()
        agg.columns = [cat, 'count', 'fraud_count', 'fraud_rate']
        max_lift = agg['fraud_rate'].max() / val_fraud_rate
        min_lift = agg['fraud_rate'].min() / val_fraud_rate
        content += f"### {cat}\n"
        content += f"Max fraud rate: {agg['fraud_rate'].max():.5f} ({max_lift:.3f}x) | Min: {agg['fraud_rate'].min():.5f} ({min_lift:.3f}x)\n\n"
        content += "| Category | Count | Fraud Rate | Lift |\n|---|---|---|---|\n"
        for _, r in agg.iterrows():
            lift = r['fraud_rate'] / val_fraud_rate
            content += f"| {r[cat]} | {r['count']:,} | {r['fraud_rate']*100:.4f}% | {lift:.4f}x |\n"
        content += "\n"

    content += "**Verdict**: If max lift ≈ min lift ≈ 1.0x across all categories, labels are uniformly distributed (synthetic random data).\n\n"
    write_md(content)

def main():
    write_md("# SANITY CHECK REPORT — Feature Discovery Pre-Validation\n", 'w')
    write_md("## A. Objective\nVerify whether dataset contains any real predictive signal before committing to expensive feature engineering.\nOnly `train.csv` used. `test.csv` untouched.\n")

    train, val = load_and_prepare()

    run_fraud_rate_uniformity_check(train)
    run_single_feature_diagnostics(train, val)
    pr_real, pr_shuffle, gap_pct = run_label_permutation_test(train, val)

    content = """
## L. Final Recommendation (After Sanity Check)

Based on evidence collected:

**Fraud Rate Uniformity**: If all categories have lift ≈ 1.0x, categorical features carry no signal.

**Score Distributions**:
- `velocity_score`: Perfectly uniform 1–20 distribution → generated randomly
- `spending_deviation_score`: Standard Normal → generated randomly
- `geo_anomaly_score`: Uniform [0,1] → generated randomly

**Device/IP Cardinality**:
- `ip_address`: ~4M unique on 4M rows → 1 IP per transaction → zero historical signal possible
- `device_hash`: 3.2M unique → too sparse for historical features

**Label Permutation Gap**: See section K above.

If gap < 5%: **STOP MODELING**. Proceed with feature discovery only as academic exercise.
If gap 5–20%: Weak signal. Feature engineering may help marginally.
If gap > 20%: Real signal. Aggressive feature engineering warranted.
"""
    write_md(content)
    print(f"\nSanity Check Completed!")
    print(f"PR-AUC (Real): {pr_real:.5f}")
    print(f"PR-AUC (Shuffle): {pr_shuffle:.5f}")
    print(f"Gap: {gap_pct:+.1f}%")

if __name__ == "__main__":
    main()
