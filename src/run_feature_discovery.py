"""
Feature Discovery & Strong Feature Mining
==========================================
Tập trung vào các feature P0:
- Novelty (is_new_receiver_for_sender, is_new_device_for_sender, ...)
- All-past frequency (sender_seen_count, receiver_in_degree, ...)
- Sender-relative amount (zscore, is_new_max, ...)
- Lifecycle (sender_tx_index, sender_days_since_first_seen, ...)
- Unusual hour (deviation from sender mean hour)

Kỹ thuật: cumcount() groupby — KHÔNG dùng rolling window để tránh duplicate-timestamp error.
Anti-leakage: cumcount() tại row i trong group = số rows đã xảy ra TRƯỚC row đó trong cùng group.

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

REPORT_PATH = "reports/feature_discovery_report.md"
TRAIN_PATH = "data/split/train.csv"

def write_md(content, mode='a'):
    with open(REPORT_PATH, mode, encoding='utf-8') as f:
        f.write(content + "\n")

def get_top_k(probs, labels_arr, pct):
    k = max(1, int(len(probs) * pct))
    idx = np.argsort(probs)[::-1][:k]
    fraud_cap = labels_arr[idx].sum()
    prec = fraud_cap / k
    rec = fraud_cap / labels_arr.sum() if labels_arr.sum() > 0 else 0
    return prec, rec

def load_data():
    print("Loading data...")
    df = pd.read_csv(TRAIN_PATH, parse_dates=['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df

def add_base_features(df):
    df['log_amount'] = np.log1p(df['amount'])
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_night'] = ((df['hour'] >= 0) & (df['hour'] <= 5)).astype(int)
    df['tslt_is_missing'] = df['time_since_last_transaction'].isnull().astype(int)
    df['tslt_is_negative'] = (df['time_since_last_transaction'] < 0).astype(int)
    df['tslt_abs'] = df['time_since_last_transaction'].abs()
    return df

# ============================================================
# NOVELTY & ALL-PAST FREQUENCY FEATURES
# All using cumcount() on sorted [entity, timestamp] → strictly past
# ============================================================

def add_novelty_and_allpast_features(df):
    print("  Building novelty & all-past frequency features...")
    created = []

    # --- Sender tx index (0-based cumcount within sender, strictly past) ---
    df_s = df[['sender_account', 'timestamp']].copy()
    df_s['sender_tx_index'] = df_s.sort_values(['sender_account', 'timestamp']) \
                                   .groupby('sender_account').cumcount()
    df['sender_tx_index'] = df_s['sender_tx_index'].values
    df['is_sender_first_tx'] = (df['sender_tx_index'] == 0).astype(int)
    df['sender_seen_count_all_past'] = df['sender_tx_index']  # alias for clarity
    created += ['sender_tx_index', 'is_sender_first_tx', 'sender_seen_count_all_past']

    # --- Receiver tx index ---
    df_r = df[['receiver_account', 'timestamp']].copy()
    df_r['receiver_tx_index'] = df_r.sort_values(['receiver_account', 'timestamp']) \
                                     .groupby('receiver_account').cumcount()
    df['receiver_tx_index'] = df_r['receiver_tx_index'].values
    df['is_receiver_first_tx'] = (df['receiver_tx_index'] == 0).astype(int)
    df['receiver_seen_count_all_past'] = df['receiver_tx_index']
    created += ['receiver_tx_index', 'is_receiver_first_tx', 'receiver_seen_count_all_past']

    # --- Sender-receiver pair novelty ---
    df_sr = df[['sender_account', 'receiver_account', 'timestamp']].copy()
    df_sr['pair_count'] = df_sr.sort_values(['sender_account', 'receiver_account', 'timestamp']) \
                                .groupby(['sender_account', 'receiver_account']).cumcount()
    df['sender_receiver_pair_count_past'] = df_sr['pair_count'].values
    df['is_new_receiver_for_sender'] = (df['sender_receiver_pair_count_past'] == 0).astype(int)
    created += ['sender_receiver_pair_count_past', 'is_new_receiver_for_sender']

    # --- Sender-device pair novelty ---
    df_sd = df[['sender_account', 'device_hash', 'timestamp']].copy()
    df_sd['pair_count'] = df_sd.sort_values(['sender_account', 'device_hash', 'timestamp']) \
                                .groupby(['sender_account', 'device_hash']).cumcount()
    df['sender_device_pair_count_past'] = df_sd['pair_count'].values
    df['is_new_device_for_sender'] = (df['sender_device_pair_count_past'] == 0).astype(int)
    created += ['sender_device_pair_count_past', 'is_new_device_for_sender']

    # --- Sender-merchant_category novelty ---
    df_sm = df[['sender_account', 'merchant_category', 'timestamp']].copy()
    df_sm['pair_count'] = df_sm.sort_values(['sender_account', 'merchant_category', 'timestamp']) \
                                .groupby(['sender_account', 'merchant_category']).cumcount()
    df['is_new_merchant_category_for_sender'] = (df_sm['pair_count'].values == 0).astype(int)
    created.append('is_new_merchant_category_for_sender')

    # --- Sender-payment_channel novelty ---
    df_sc = df[['sender_account', 'payment_channel', 'timestamp']].copy()
    df_sc['pair_count'] = df_sc.sort_values(['sender_account', 'payment_channel', 'timestamp']) \
                                .groupby(['sender_account', 'payment_channel']).cumcount()
    df['is_new_payment_channel_for_sender'] = (df_sc['pair_count'].values == 0).astype(int)
    created.append('is_new_payment_channel_for_sender')

    # --- Sender-location novelty ---
    df_sl = df[['sender_account', 'location', 'timestamp']].copy()
    df_sl['pair_count'] = df_sl.sort_values(['sender_account', 'location', 'timestamp']) \
                                .groupby(['sender_account', 'location']).cumcount()
    df['is_new_location_for_sender'] = (df_sl['pair_count'].values == 0).astype(int)
    created.append('is_new_location_for_sender')

    # --- Receiver in-degree (unique senders, all past) ---
    df_ri = df[['receiver_account', 'sender_account', 'timestamp']].copy()
    df_ri['unique_sender_count'] = df_ri.sort_values(['receiver_account', 'sender_account', 'timestamp']) \
                                         .groupby(['receiver_account', 'sender_account']).cumcount()
    # Mark first time a sender-receiver pair appears, then count distinct senders per receiver cumulatively
    df_ri['is_new_sender_for_receiver'] = (df_ri['unique_sender_count'] == 0).astype(int)
    df_ri_sorted = df_ri.sort_values(['receiver_account', 'timestamp'])
    df_ri_sorted['receiver_in_degree_all_past'] = df_ri_sorted.groupby('receiver_account')['is_new_sender_for_receiver'].cumsum() - df_ri_sorted['is_new_sender_for_receiver']
    df_ri_sorted = df_ri_sorted.sort_index()
    df['receiver_in_degree_all_past'] = df_ri_sorted['receiver_in_degree_all_past'].values
    df['is_new_sender_for_receiver'] = df_ri_sorted['is_new_sender_for_receiver'].values
    created += ['receiver_in_degree_all_past', 'is_new_sender_for_receiver']

    # --- Sender out-degree (unique receivers, all past) ---
    df_so = df[['sender_account', 'receiver_account', 'timestamp']].copy()
    df_so['is_new_rcv'] = (df_so.sort_values(['sender_account', 'receiver_account', 'timestamp'])
                               .groupby(['sender_account', 'receiver_account']).cumcount() == 0).astype(int)
    df_so_sorted = df_so.sort_values(['sender_account', 'timestamp'])
    df_so_sorted['sender_out_degree_all_past'] = df_so_sorted.groupby('sender_account')['is_new_rcv'].cumsum() - df_so_sorted['is_new_rcv']
    df_so_sorted = df_so_sorted.sort_index()
    df['sender_out_degree_all_past'] = df_so_sorted['sender_out_degree_all_past'].values
    created.append('sender_out_degree_all_past')

    print(f"    Created: {created}")
    return df, created

# ============================================================
# SENDER-RELATIVE AMOUNT FEATURES
# Expanding mean/std per sender, then compute zscore
# ============================================================

def add_sender_amount_features(df):
    print("  Building sender-relative amount features...")
    created = []

    df_sa = df[['sender_account', 'timestamp', 'amount']].copy()
    df_sa = df_sa.sort_values(['sender_account', 'timestamp'])

    # Expanding cumulative sum and count (strictly past = shift by 1)
    df_sa['cum_sum'] = df_sa.groupby('sender_account')['amount'].transform(
        lambda x: x.shift(1).expanding().sum()
    )
    df_sa['cum_count'] = df_sa.groupby('sender_account')['amount'].transform(
        lambda x: x.shift(1).expanding().count()
    )
    df_sa['cum_sum2'] = df_sa.groupby('sender_account')['amount'].transform(
        lambda x: (x ** 2).shift(1).expanding().sum()
    )

    df_sa['sender_amount_mean_past'] = df_sa['cum_sum'] / df_sa['cum_count']
    df_sa['sender_amount_std_past'] = np.sqrt(
        (df_sa['cum_sum2'] / df_sa['cum_count']) - df_sa['sender_amount_mean_past'] ** 2
    ).clip(lower=0)
    df_sa['sender_amount_max_past'] = df_sa.groupby('sender_account')['amount'].transform(
        lambda x: x.shift(1).expanding().max()
    )

    df_sa = df_sa.sort_index()

    df['sender_amount_mean_past'] = df_sa['sender_amount_mean_past'].values
    df['sender_amount_std_past'] = df_sa['sender_amount_std_past'].values
    df['sender_amount_max_past'] = df_sa['sender_amount_max_past'].values

    # Z-score: (amount - mean) / (std + epsilon)
    df['sender_amount_zscore_past'] = (
        (df['amount'] - df['sender_amount_mean_past']) /
        (df['sender_amount_std_past'] + 1e-6)
    )

    # Is new max amount
    df['is_sender_new_max_amount'] = (
        (df['amount'] > df['sender_amount_max_past'].fillna(-np.inf))
    ).astype(int)

    # Ratio to mean
    df['amount_ratio_to_sender_mean'] = df['amount'] / (df['sender_amount_mean_past'] + 1e-6)

    created += ['sender_amount_mean_past', 'sender_amount_std_past', 'sender_amount_max_past',
                'sender_amount_zscore_past', 'is_sender_new_max_amount', 'amount_ratio_to_sender_mean']

    print(f"    Created: {created}")
    return df, created

# ============================================================
# LIFECYCLE FEATURES
# Days since first seen, since last seen, dormancy
# ============================================================

def add_lifecycle_features(df):
    print("  Building lifecycle features...")
    created = []

    df_lc = df[['sender_account', 'receiver_account', 'timestamp']].copy()
    df_lc = df_lc.sort_values(['sender_account', 'timestamp'])

    # Sender: days since first seen
    df_lc['sender_first_ts'] = df_lc.groupby('sender_account')['timestamp'].transform('min')
    df_lc['sender_days_since_first_seen'] = (df_lc['timestamp'] - df_lc['sender_first_ts']).dt.total_seconds() / 86400

    # Sender: days since last seen (strictly previous tx in same sender group)
    df_lc['sender_prev_ts'] = df_lc.groupby('sender_account')['timestamp'].shift(1)
    df_lc['sender_days_since_last_seen'] = (df_lc['timestamp'] - df_lc['sender_prev_ts']).dt.total_seconds() / 86400

    df_lc = df_lc.sort_index()
    df['sender_days_since_first_seen'] = df_lc['sender_days_since_first_seen'].values
    df['sender_days_since_last_seen'] = df_lc['sender_days_since_last_seen'].values

    # Receiver: days since first seen
    df_r = df[['receiver_account', 'timestamp']].copy()
    df_r = df_r.sort_values(['receiver_account', 'timestamp'])
    df_r['receiver_first_ts'] = df_r.groupby('receiver_account')['timestamp'].transform('min')
    df_r['receiver_days_since_first_seen'] = (df_r['timestamp'] - df_r['receiver_first_ts']).dt.total_seconds() / 86400
    df_r = df_r.sort_index()
    df['receiver_days_since_first_seen'] = df_r['receiver_days_since_first_seen'].values

    created += ['sender_days_since_first_seen', 'sender_days_since_last_seen', 'receiver_days_since_first_seen']
    print(f"    Created: {created}")
    return df, created

# ============================================================
# UNUSUAL HOUR FEATURES
# Sender's historical mean hour vs current hour
# ============================================================

def add_unusual_hour_features(df):
    print("  Building unusual-hour features...")
    created = []

    df_h = df[['sender_account', 'timestamp', 'hour']].copy()
    df_h = df_h.sort_values(['sender_account', 'timestamp'])

    df_h['cum_hour_sum'] = df_h.groupby('sender_account')['hour'].transform(
        lambda x: x.shift(1).expanding().sum()
    )
    df_h['cum_hour_count'] = df_h.groupby('sender_account')['hour'].transform(
        lambda x: x.shift(1).expanding().count()
    )
    df_h['sender_mean_hour_past'] = df_h['cum_hour_sum'] / df_h['cum_hour_count']
    df_h['hour_deviation_from_sender_mean'] = (df_h['hour'] - df_h['sender_mean_hour_past']).abs()
    df_h = df_h.sort_index()

    df['sender_mean_hour_past'] = df_h['sender_mean_hour_past'].values
    df['hour_deviation_from_sender_mean'] = df_h['hour_deviation_from_sender_mean'].values
    df['is_unusual_hour_for_sender'] = (df['hour_deviation_from_sender_mean'] > 6).astype(int)

    created += ['sender_mean_hour_past', 'hour_deviation_from_sender_mean', 'is_unusual_hour_for_sender']
    print(f"    Created: {created}")
    return df, created

# ============================================================
# FEATURE DIAGNOSTICS
# ============================================================

def feature_diagnostics(df, new_features, val_mask):
    print("  Running feature diagnostics...")
    train_mask = ~val_mask
    val_df = df[val_mask]
    y_val = val_df['is_fraud'].astype(int).values
    val_fraud_rate = y_val.mean()

    rows = []
    for feat in new_features:
        if feat not in df.columns:
            continue
        col = df.loc[val_mask, feat]
        null_rate = col.isnull().mean()
        zero_rate = (col.fillna(0) == 0).mean()

        try:
            col_filled = col.fillna(col.median() if col.dtype in [np.float64, np.float32, np.int64] else 0)
            pr_pos = average_precision_score(y_val, col_filled)
            pr_neg = average_precision_score(y_val, -col_filled)
            pr = max(pr_pos, pr_neg)
            rel_lift = pr / val_fraud_rate
        except Exception:
            pr = None
            rel_lift = None

        rows.append({
            'Feature': feat,
            'Null Rate': null_rate,
            'Zero Rate': zero_rate,
            'PR-AUC': pr,
            'Rel Lift': rel_lift
        })

    return rows

# ============================================================
# MODELING EXPERIMENTS
# ============================================================

def run_experiment(name, X_train, y_train, X_val, y_val, num_cols, cat_cols):
    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median'))])
    cat_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                                      ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ])

    imbalance = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
    model = LGBMClassifier(scale_pos_weight=imbalance, n_estimators=100, random_state=42, n_jobs=-1)
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])

    print(f"  Training {name}...")
    pipeline.fit(X_train, y_train)

    probs_val = pipeline.predict_proba(X_val)[:, 1]
    probs_train = pipeline.predict_proba(X_train)[:, 1]

    pr_val = average_precision_score(y_val, probs_val)
    pr_train = average_precision_score(y_train, probs_train)
    roc = roc_auc_score(y_val, probs_val)

    y_val_arr = np.array(y_val)
    p1, r1 = get_top_k(probs_val, y_val_arr, 0.01)
    p5, r5 = get_top_k(probs_val, y_val_arr, 0.05)

    # Feature importance
    try:
        classifier = pipeline.named_steps['classifier']
        prep = pipeline.named_steps['preprocessor']
        ohe = prep.named_transformers_['cat'].named_steps['onehot']
        ohe_names = ohe.get_feature_names_out(cat_cols)
        all_names = num_cols + list(ohe_names)
        imp = pd.DataFrame({'Feature': all_names, 'Importance': classifier.feature_importances_})
        imp = imp.sort_values('Importance', ascending=False)
    except Exception:
        imp = pd.DataFrame()

    return {
        'name': name,
        'pr_val': pr_val,
        'pr_train': pr_train,
        'gap': pr_train - pr_val,
        'roc': roc,
        'p1': p1, 'r1': r1,
        'p5': p5, 'r5': r5,
        'probs_val': probs_val,
        'importance': imp
    }

def run_all_experiments(df, all_new_feats):
    train = df.iloc[:3200000]
    val = df.iloc[3200000:4000000]
    y_train = train['is_fraud'].astype(int)
    y_val = val['is_fraud'].astype(int)
    val_fraud_rate = y_val.mean()

    base_num = ['amount', 'log_amount', 'spending_deviation_score', 'velocity_score',
                'geo_anomaly_score', 'hour', 'day_of_week', 'month',
                'is_weekend', 'is_night', 'tslt_abs', 'tslt_is_missing', 'tslt_is_negative']
    cat_cols = ['transaction_type', 'merchant_category', 'payment_channel', 'device_used', 'location']

    novelty_feats = [f for f in all_new_feats if 'is_new' in f or 'pair_count' in f or 'tx_index' in f or 'seen_count' in f or 'degree' in f or 'first_tx' in f]
    sender_amt_feats = [f for f in all_new_feats if 'amount_mean' in f or 'amount_std' in f or 'amount_max' in f or 'zscore' in f or 'ratio' in f or 'new_max' in f]
    lifecycle_feats = [f for f in all_new_feats if 'days_since' in f or 'lifecycle' in f]
    hour_feats = [f for f in all_new_feats if 'hour' in f and f not in base_num]
    graph_feats = [f for f in all_new_feats if 'degree' in f]

    experiments = [
        ('Exp 0 — Baseline', base_num, cat_cols),
        ('Exp 1 — Baseline + Novelty', base_num + novelty_feats, cat_cols),
        ('Exp 2 — Baseline + Sender Amount', base_num + sender_amt_feats, cat_cols),
        ('Exp 3 — Baseline + Lifecycle', base_num + lifecycle_feats, cat_cols),
        ('Exp 4 — Baseline + Unusual Hour', base_num + hour_feats, cat_cols),
        ('Exp 5 — All New Features', base_num + all_new_feats, cat_cols),
    ]

    results = []
    for exp_name, num_cols, c_cols in experiments:
        valid_num = [c for c in num_cols if c in train.columns]
        valid_cat = [c for c in c_cols if c in train.columns]
        try:
            res = run_experiment(
                exp_name,
                train[valid_num + valid_cat], y_train,
                val[valid_num + valid_cat], y_val,
                valid_num, valid_cat
            )
            results.append(res)
        except Exception as e:
            print(f"  Error in {exp_name}: {e}")

    # Exp 6: Hard subset
    try:
        train_hard = train[train['tslt_is_missing'] == 0].copy()
        val_hard = val[val['tslt_is_missing'] == 0].copy()
        hard_num = [c for c in base_num + all_new_feats if c != 'tslt_is_missing' and c in train_hard.columns]
        hard_cat = [c for c in cat_cols if c in train_hard.columns]
        res = run_experiment(
            'Exp 6 — Hard Subset + All Features',
            train_hard[hard_num + hard_cat], train_hard['is_fraud'].astype(int),
            val_hard[hard_num + hard_cat], val_hard['is_fraud'].astype(int),
            hard_num, hard_cat
        )
        results.append(res)
    except Exception as e:
        print(f"  Error in hard subset: {e}")

    # Time stability for best model
    best = max(results, key=lambda r: r['pr_val'])
    probs_best = best['probs_val']
    val_blocks = []
    chunk = len(val) // 4
    for i in range(4):
        s, e = i * chunk, (i+1)*chunk if i < 3 else len(val)
        y_blk = y_val.iloc[s:e].values
        p_blk = probs_best[s:e]
        try:
            pr_blk = average_precision_score(y_blk, p_blk)
            p1_blk, _ = get_top_k(p_blk, y_blk, 0.01)
            val_blocks.append({
                'block': i+1,
                'fraud_rate': y_blk.mean(),
                'pr_auc': pr_blk,
                'prec_1pct': p1_blk
            })
        except Exception:
            pass

    return results, val_blocks, val_fraud_rate

def write_report(results, val_blocks, val_fraud_rate, diag_rows, all_new_feats):
    write_md("## G. New Feature Diagnostics\n")
    write_md("| Feature | Null Rate | Zero Rate | PR-AUC | Rel Lift | Signal |\n|---|---|---|---|---|---|")
    for r in diag_rows:
        pr_str = f"{r['PR-AUC']:.5f}" if r['PR-AUC'] is not None else "N/A"
        rl_str = f"{r['Rel Lift']:.3f}x" if r['Rel Lift'] is not None else "N/A"
        signal = "✅" if (r['Rel Lift'] or 0) > 1.1 else ("⚠️" if (r['Rel Lift'] or 0) > 1.02 else "❌")
        write_md(f"| {r['Feature']} | {r['Null Rate']:.2%} | {r['Zero Rate']:.2%} | {pr_str} | {rl_str} | {signal} |")

    write_md("\n## H. Modeling Experiment Results\n")
    write_md("| Experiment | Train PR-AUC | Val PR-AUC | Gap | Diff vs Baseline | Rel Lift | Prec@1% | Prec@5% |\n|---|---|---|---|---|---|---|---|")
    baseline_pr = next((r['pr_val'] for r in results if 'Baseline' in r['name'] and 'Exp 0' in r['name']), None)
    for r in results:
        diff = r['pr_val'] - baseline_pr if baseline_pr else 0
        rl = r['pr_val'] / val_fraud_rate
        write_md(f"| {r['name']} | {r['pr_train']:.5f} | {r['pr_val']:.5f} | {r['gap']:+.5f} | {diff:+.5f} | {rl:.3f}x | {r['p1']*100:.2f}% | {r['p5']*100:.2f}% |")

    write_md("\n## J. Time Stability Review\n")
    write_md("| Block | Fraud Rate | PR-AUC | Prec@1% |\n|---|---|---|---|")
    for b in val_blocks:
        write_md(f"| {b['block']} | {b['fraud_rate']*100:.4f}% | {b['pr_auc']:.5f} | {b['prec_1pct']*100:.2f}% |")

    # Feature importance of best model
    best = max(results, key=lambda r: r['pr_val'])
    if not best['importance'].empty:
        write_md("\n## Top 20 Features (Best Experiment)\n")
        write_md("| Rank | Feature | Importance |\n|---|---|---|")
        for i, row in best['importance'].head(20).reset_index().iterrows():
            write_md(f"| {i+1} | {row['Feature']} | {row['Importance']:.0f} |")

    best_pr = max(r['pr_val'] for r in results)
    improved = best_pr > (baseline_pr or 0) * 1.05

    write_md(f"""
## L. Final Recommendation

### Key Metrics
- **Random Baseline PR-AUC**: {val_fraud_rate:.5f}
- **Previous Baseline PR-AUC**: ~0.04454
- **Best New Experiment PR-AUC**: {best_pr:.5f}
- **Best Experiment**: {best['name']}
- **Improvement over previous baseline**: {(best_pr - 0.04454):+.5f}

### Conclusions

1. **Tìm được feature mạnh mới không?** {"Yes — improvement recorded." if improved else "No — no meaningful improvement found."}
2. **PR-AUC vượt rõ baseline không?** {"Yes" if best_pr > 0.04454 * 1.1 else "Marginal or No"}
3. **Có overfit không?** {"Possible" if best['gap'] > 0.01 else "Minimal"}
4. **Có nên tune model chưa?** {"Consider tuning if signal is confirmed." if improved else "DO NOT TUNE — no signal to tune."}
5. **Có được mở test.csv chưa?** **NO — test.csv remains locked.**

### Dataset Signal Verdict
{"🟡 Some improvement detected. Investigate further before tuning." if improved else "🔴 CONFIRMED: Dataset has zero/negligible predictive signal beyond the tslt_is_missing artifact. Recommend stopping modeling and requesting new data or additional feature sources."}
""")

def main():
    write_md("# FEATURE DISCOVERY & STRONG FEATURE MINING REPORT\n", 'w')
    write_md("## A. Objective\nDiscover whether genuine predictive signal exists in this dataset beyond the tslt_is_missing artifact, using novelty, all-past frequency, sender-relative amount, lifecycle, and unusual-hour features.\nOnly `train.csv` used. `test.csv` untouched.\n")

    df = load_data()
    df = add_base_features(df)

    write_md("## E. Selected P0/P1 Features\nBuilding the following feature families:\n- Novelty: is_new_receiver_for_sender, is_new_device_for_sender, is_new_merchant_category_for_sender, is_new_payment_channel_for_sender, is_new_location_for_sender\n- All-past frequency: sender_tx_index, receiver_tx_index, sender_seen_count_all_past, receiver_seen_count_all_past, receiver_in_degree_all_past, sender_out_degree_all_past\n- Sender-relative amount: sender_amount_zscore_past, is_sender_new_max_amount, amount_ratio_to_sender_mean\n- Lifecycle: sender_days_since_first_seen, sender_days_since_last_seen, receiver_days_since_first_seen\n- Unusual-hour: hour_deviation_from_sender_mean, is_unusual_hour_for_sender\n\nIP/device rolling features REJECTED due to near-unique cardinality (>3M unique values on 4M rows).\n")

    write_md("## F. Leakage Safety Design\n- All features use `cumcount()` or expanding transform with `shift(1)`, strictly counting only rows BEFORE the current row in chronological order within each entity group.\n- No `is_fraud` label used in feature computation.\n- Preprocessing (Imputer) fit strictly on train_inner (first 3.2M rows), transform on validation.\n")

    all_new_feats = []
    df, created = add_novelty_and_allpast_features(df)
    all_new_feats.extend(created)

    df, created = add_sender_amount_features(df)
    all_new_feats.extend(created)

    df, created = add_lifecycle_features(df)
    all_new_feats.extend(created)

    df, created = add_unusual_hour_features(df)
    all_new_feats.extend(created)

    print(f"Total new features created: {len(all_new_feats)}")

    # Diagnostics on validation
    val_mask = df.index >= 3200000
    diag_rows = feature_diagnostics(df, all_new_feats, val_mask)

    results, val_blocks, val_fraud_rate = run_all_experiments(df, all_new_feats)
    write_report(results, val_blocks, val_fraud_rate, diag_rows, all_new_feats)

    print("Feature Discovery Completed!")

if __name__ == "__main__":
    main()
