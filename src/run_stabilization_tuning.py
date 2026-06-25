"""
Candidate Feature Stabilization & Controlled LightGBM Tuning
=============================================================
Steps:
  1. Feature Set Comparison (FS0–FS5) with fixed LightGBM baseline
  2. Group Ablation (7 groups)
  3. Controlled LightGBM Tuning (small grid, early stopping)
  4. Threshold & Top-K Analysis
  5. Time Stability (4 blocks)
  6. Sanity Checks (label permutation, with/without TSLT, hard subset)

Only train.csv used. test.csv untouched.
OOT split: train_inner = first 3.2M, validation = last 800K.
"""

import pandas as pd
import numpy as np
from itertools import product as iterproduct
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             precision_recall_curve, f1_score,
                             confusion_matrix)
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
import warnings
warnings.filterwarnings('ignore')

REPORT_PATH = "reports/stabilization_tuning_report.md"
TRAIN_PATH = "data/split/train.csv"

def wmd(content, mode='a'):
    with open(REPORT_PATH, mode, encoding='utf-8') as f:
        f.write(content + "\n")

def get_topk(probs, labels, pct):
    k = max(1, int(len(probs) * pct))
    idx = np.argsort(probs)[::-1][:k]
    fraud = labels[idx].sum()
    return fraud / k, fraud / max(labels.sum(), 1), fraud, k

# ============================================================
# DATA LOADING & FEATURE BUILDING
# ============================================================

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

def add_novelty_features(df):
    print("  Building novelty & familiarity features...")
    # Pair count features (cumcount on sorted [sender, category, timestamp])
    pair_configs = [
        ('sender_account', 'location',          'sender_location_pair_count_past',       'is_new_location_for_sender'),
        ('sender_account', 'payment_channel',   'sender_channel_pair_count_past',        'is_new_payment_channel_for_sender'),
        ('sender_account', 'device_used',       'sender_device_type_pair_count_past',    'is_new_device_used_for_sender'),
        ('sender_account', 'transaction_type',  'sender_txn_type_pair_count_past',       'is_new_transaction_type_for_sender'),
        ('sender_account', 'merchant_category', 'sender_category_pair_count_past',       'is_new_merchant_category_for_sender'),
    ]
    for s_col, v_col, count_feat, novelty_feat in pair_configs:
        df2 = df[[s_col, v_col, 'timestamp']].copy()
        df2['oi'] = df2.index
        df2 = df2.sort_values([s_col, v_col, 'timestamp'])
        df2['pc'] = df2.groupby([s_col, v_col]).cumcount()
        df2 = df2.sort_values('oi')
        df[count_feat] = df2['pc'].values
        df[novelty_feat] = (df[count_feat] == 0).astype(int)

    # Sender tx index (cumcount within sender)
    df2 = df[['sender_account', 'timestamp']].copy()
    df2['oi'] = df2.index
    df2 = df2.sort_values(['sender_account', 'timestamp'])
    df2['tidx'] = df2.groupby('sender_account').cumcount()
    df2 = df2.sort_values('oi')
    df['sender_tx_index'] = df2['tidx'].values
    df['sender_seen_count_all_past'] = df['sender_tx_index']

    # Cumulative unique counts per sender
    def cum_unique(df, s_col, v_col):
        df2 = df[[s_col, v_col, 'timestamp']].copy()
        df2['oi'] = df2.index
        df2 = df2.sort_values([s_col, v_col, 'timestamp'])
        df2['is_first'] = (df2.groupby([s_col, v_col]).cumcount() == 0).astype(int)
        df2 = df2.sort_values([s_col, 'timestamp'])
        df2['cum_u'] = df2.groupby(s_col)['is_first'].cumsum() - df2['is_first']
        df2 = df2.sort_values('oi')
        return df2['cum_u'].values

    df['sender_unique_receivers_all_past']  = cum_unique(df, 'sender_account', 'receiver_account')
    df['sender_unique_locations_all_past']  = cum_unique(df, 'sender_account', 'location')
    df['sender_unique_channels_all_past']   = cum_unique(df, 'sender_account', 'payment_channel')
    df['sender_unique_categories_all_past'] = cum_unique(df, 'sender_account', 'merchant_category')
    df['sender_unique_device_types_all_past'] = cum_unique(df, 'sender_account', 'device_used')
    df['sender_out_degree_all_past'] = df['sender_unique_receivers_all_past']
    print("  Done.")
    return df

# ============================================================
# PREPROCESSING & TRAINING HELPERS
# ============================================================

CAT_COLS = ['transaction_type', 'merchant_category', 'payment_channel', 'device_used', 'location']

def build_preprocessor(num_cols):
    nt = Pipeline([('imp', SimpleImputer(strategy='median'))])
    ct = Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                   ('ohe', OneHotEncoder(handle_unknown='ignore'))])
    return ColumnTransformer([('n', nt, num_cols), ('c', ct, CAT_COLS)])

def train_eval(X_tr, y_tr, X_va, y_va, num_cols, lgbm_params=None, early_stop=False, name=""):
    pre = build_preprocessor(num_cols)
    if lgbm_params is None:
        imb = (len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1)
        lgbm_params = dict(scale_pos_weight=imb, n_estimators=100, random_state=42, n_jobs=-1)
    mdl = LGBMClassifier(**lgbm_params)

    if early_stop and 'n_estimators' in lgbm_params and lgbm_params['n_estimators'] > 200:
        # Fit preprocessor separately for early stopping
        X_tr_pre = pre.fit_transform(X_tr)
        X_va_pre = pre.transform(X_va)
        mdl.fit(X_tr_pre, y_tr,
                eval_set=[(X_va_pre, y_va)],
                eval_metric='average_precision',
                callbacks=[early_stopping(50, verbose=False), log_evaluation(-1)])
        pv = mdl.predict_proba(X_va_pre)[:, 1]
        pt = mdl.predict_proba(X_tr_pre)[:, 1]
        imp_names = None
    else:
        pipe = Pipeline([('pre', pre), ('mdl', mdl)])
        pipe.fit(X_tr, y_tr)
        pv = pipe.predict_proba(X_va)[:, 1]
        pt = pipe.predict_proba(X_tr)[:, 1]
        try:
            ohe = pipe['pre'].named_transformers_['c']['ohe']
            ohe_names = ohe.get_feature_names_out(CAT_COLS)
            imp_names = num_cols + list(ohe_names)
        except Exception:
            imp_names = None
        mdl = pipe['mdl']

    pr_v = average_precision_score(y_va, pv)
    pr_t = average_precision_score(y_tr, pt)
    roc   = roc_auc_score(y_va, pv)
    ya = np.array(y_va)
    p001, r001, _, k001 = get_topk(pv, ya, 0.001)
    p1,   r1,   f1c, k1  = get_topk(pv, ya, 0.01)
    p5,   r5,   _, _     = get_topk(pv, ya, 0.05)

    imp = pd.DataFrame()
    if imp_names is not None:
        try:
            imp = pd.DataFrame({'Feature': imp_names, 'Importance': mdl.feature_importances_})
            imp = imp.sort_values('Importance', ascending=False)
        except Exception:
            pass

    if name:
        print(f"  [{name}] Val PR-AUC={pr_v:.5f}  Train={pr_t:.5f}  Gap={pr_t-pr_v:+.5f}")
    return dict(pr_v=pr_v, pr_t=pr_t, gap=pr_t-pr_v, roc=roc,
                p001=p001, r001=r001, p1=p1, r1=r1, p5=p5, r5=r5,
                probs=pv, imp=imp, n_best=getattr(mdl, 'best_iteration_', lgbm_params.get('n_estimators',100)))

def block_stats(probs, y_va_arr, n=4):
    chunk = len(y_va_arr) // n
    rows = []
    for i in range(n):
        s, e = i*chunk, (i+1)*chunk if i < n-1 else len(y_va_arr)
        yb, pb = y_va_arr[s:e], probs[s:e]
        try:
            pr = average_precision_score(yb, pb)
            p1, _, _, _ = get_topk(pb, yb, 0.01)
            p5, _, _, _ = get_topk(pb, yb, 0.05)
            r1_b = get_topk(pb, yb, 0.01)[1]
            rows.append(dict(block=i+1, fr=yb.mean(), pr=pr, p1=p1, p5=p5, r1=r1_b))
        except Exception:
            rows.append(dict(block=i+1, fr=yb.mean(), pr=None, p1=None, p5=None, r1=None))
    return rows

# ============================================================
# MAIN
# ============================================================

def main():
    wmd("# CANDIDATE FEATURE STABILIZATION & CONTROLLED TUNING REPORT\n", 'w')
    wmd("## A. Objective\nStabilize best feature set from novelty/pair-familiarity round, prune noise, confirm improvement is real, then perform controlled LightGBM tuning.\n")
    wmd("## B. Confirmation\n- Only `train.csv` used. `test.csv` untouched.\n- OOT split: train_inner = 3.2M rows, validation = 800K rows.\n- No fraud label in feature computation. No target encoding. No SMOTE.\n")

    df = load_data()
    df = add_base_features(df)
    df = add_novelty_features(df)

    train = df.iloc[:3200000]
    val   = df.iloc[3200000:4000000]
    y_tr  = train['is_fraud'].astype(int)
    y_va  = val['is_fraud'].astype(int)
    ya    = np.array(y_va)
    vfr   = ya.mean()

    # ---- Feature Sets ----
    BASE_NUM = ['amount', 'log_amount', 'spending_deviation_score', 'velocity_score',
                'geo_anomaly_score', 'hour', 'day_of_week', 'month',
                'is_weekend', 'is_night', 'tslt_abs', 'tslt_is_missing', 'tslt_is_negative']

    NOVELTY_BINARY = ['is_new_location_for_sender', 'is_new_payment_channel_for_sender',
                      'is_new_transaction_type_for_sender', 'is_new_device_used_for_sender']

    PAIR_COUNTS = ['sender_location_pair_count_past', 'sender_channel_pair_count_past',
                   'sender_device_type_pair_count_past', 'sender_txn_type_pair_count_past']

    SENDER_MATURITY = ['sender_tx_index', 'sender_seen_count_all_past', 'sender_out_degree_all_past',
                       'sender_unique_receivers_all_past', 'sender_unique_locations_all_past',
                       'sender_unique_channels_all_past', 'sender_unique_categories_all_past',
                       'sender_unique_device_types_all_past']

    def vc(feats): return [f for f in feats if f in df.columns]

    FS = {
        'FS0 — Baseline':              vc(BASE_NUM),
        'FS1 — Baseline + Novelty Binary': vc(BASE_NUM + NOVELTY_BINARY),
        'FS2 — Baseline + Pair Counts':    vc(BASE_NUM + PAIR_COUNTS),
        'FS3 — Baseline + Sender Maturity':vc(BASE_NUM + SENDER_MATURITY),
        'FS4 — Baseline + Best Selected':  vc(BASE_NUM + NOVELTY_BINARY + PAIR_COUNTS + SENDER_MATURITY),
        'FS5 — FS4 minus TSLT':            vc([f for f in BASE_NUM if 'tslt' not in f] + NOVELTY_BINARY + PAIR_COUNTS + SENDER_MATURITY),
    }

    # ---- Step 1: Feature Set Comparison ----
    print("\n=== Step 1: Feature Set Comparison ===")
    fs_results = {}
    for fs_name, num_cols in FS.items():
        r = train_eval(train[num_cols+CAT_COLS], y_tr, val[num_cols+CAT_COLS], y_va, num_cols, name=fs_name)
        fs_results[fs_name] = r

    wmd("## C. Feature Set Comparison (FS0–FS5)\n")
    wmd("| Feature Set | #Feats | Train PR | Val PR | Gap | Diff vs FS0 | Rel Lift | Prec@0.1% | Prec@1% | Prec@5% | Rec@1% | Rec@5% |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
    fs0_pr = fs_results['FS0 — Baseline']['pr_v']
    for fs_name, r in fs_results.items():
        nf = len(FS[fs_name])
        diff = r['pr_v'] - fs0_pr
        rl = r['pr_v'] / vfr
        wmd(f"| {fs_name} | {nf} | {r['pr_t']:.5f} | {r['pr_v']:.5f} | {r['gap']:+.5f} | {diff:+.5f} | {rl:.3f}x | {r['p001']*100:.2f}% | {r['p1']*100:.2f}% | {r['p5']*100:.2f}% | {r['r1']*100:.2f}% | {r['r5']*100:.2f}% |")

    # Best FS for ablation/tuning
    best_fs_name = max(fs_results, key=lambda k: fs_results[k]['pr_v'])
    best_fs_num  = FS[best_fs_name]
    best_fs_pr   = fs_results[best_fs_name]['pr_v']
    print(f"\nBest FS: {best_fs_name} — Val PR={best_fs_pr:.5f}")

    # ---- Step 2: Group Ablation ----
    print("\n=== Step 2: Group Ablation ===")
    ALL_NEW = vc(NOVELTY_BINARY + PAIR_COUNTS + SENDER_MATURITY)
    ALL_NUM = vc(BASE_NUM + ALL_NEW)

    TSLT_FEATS = ['tslt_abs', 'tslt_is_missing', 'tslt_is_negative']
    SCORE_FEATS = ['spending_deviation_score', 'velocity_score', 'geo_anomaly_score']
    LOWCARD_CATS = CAT_COLS

    def remove(feats, to_remove):
        return [f for f in feats if f not in to_remove]

    ablation_groups = [
        ('Full FS4',                   ALL_NUM),
        ('Remove novelty binary',      remove(ALL_NUM, NOVELTY_BINARY)),
        ('Remove pair counts',         remove(ALL_NUM, PAIR_COUNTS)),
        ('Remove sender maturity',     remove(ALL_NUM, SENDER_MATURITY)),
        ('Remove TSLT features',       remove(ALL_NUM, TSLT_FEATS)),
        ('Remove behavioral scores',   remove(ALL_NUM, SCORE_FEATS)),
    ]

    ablation_results = []
    for aname, nums in ablation_groups:
        nums = vc(nums)
        r = train_eval(train[nums+CAT_COLS], y_tr, val[nums+CAT_COLS], y_va, nums, name=aname)
        ablation_results.append((aname, r))

    wmd("\n## E. Group Ablation Results\n")
    wmd("| Group Removed | Val PR | Diff vs Full | Impact |\n|---|---|---|---|")
    full_pr = ablation_results[0][1]['pr_v']
    for aname, r in ablation_results:
        diff = r['pr_v'] - full_pr
        impact = '🔴 HIGH' if diff < -0.003 else ('🟡 MEDIUM' if diff < -0.0005 else '🟢 LOW')
        wmd(f"| {aname} | {r['pr_v']:.5f} | {diff:+.5f} | {impact} |")

    # ---- Step 3: Controlled LightGBM Tuning ----
    print("\n=== Step 3: Controlled LightGBM Tuning ===")
    # Use best feature set from FS comparison
    tune_nums = best_fs_num
    X_tr_t = train[tune_nums + CAT_COLS]
    X_va_t = val[tune_nums + CAT_COLS]

    imb_ratio = (len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1)
    sqrt_imb  = np.sqrt(imb_ratio)

    # Small grid
    grid = [
        dict(num_leaves=nl, max_depth=md, learning_rate=lr,
             n_estimators=500, min_child_samples=mcs,
             subsample=ss, colsample_bytree=cbt,
             reg_alpha=ra, reg_lambda=rl_,
             scale_pos_weight=spw, random_state=42, n_jobs=-1)
        for nl, md, lr, mcs, ss, cbt, ra, rl_, spw in iterproduct(
            [31, 63],
            [5, -1],
            [0.05, 0.1],
            [300, 1000],
            [0.9],
            [0.9],
            [0.1],
            [1, 5],
            [imb_ratio, sqrt_imb],
        )
    ]

    print(f"  Tuning grid size: {len(grid)} configs")

    # Preprocess once for early stopping
    pre_tune = build_preprocessor(tune_nums)
    X_tr_pre = pre_tune.fit_transform(X_tr_t)
    X_va_pre = pre_tune.transform(X_va_t)

    tune_results = []
    for i, params in enumerate(grid):
        mdl = LGBMClassifier(**params)
        mdl.fit(X_tr_pre, y_tr,
                eval_set=[(X_va_pre, y_va)],
                eval_metric='average_precision',
                callbacks=[early_stopping(50, verbose=False), log_evaluation(-1)])
        pv = mdl.predict_proba(X_va_pre)[:, 1]
        pt = mdl.predict_proba(X_tr_pre)[:, 1]
        pr_v = average_precision_score(y_va, pv)
        pr_t = average_precision_score(y_tr, pt)
        p1, r1, _, _ = get_topk(pv, ya, 0.01)
        p5, r5, _, _ = get_topk(pv, ya, 0.05)
        n_best = mdl.best_iteration_
        tune_results.append(dict(
            nl=params['num_leaves'], md=params['max_depth'], lr=params['learning_rate'],
            mcs=params['min_child_samples'], ra=params['reg_alpha'], rl=params['reg_lambda'],
            spw=round(params['scale_pos_weight'], 1),
            n_best=n_best, pr_v=pr_v, pr_t=pr_t, gap=pr_t-pr_v,
            p1=p1, p5=p5, probs=pv
        ))
        if (i+1) % 8 == 0:
            print(f"    {i+1}/{len(grid)} done, best so far: {max(r['pr_v'] for r in tune_results):.5f}")

    tune_results.sort(key=lambda r: r['pr_v'], reverse=True)
    best_tune = tune_results[0]

    wmd("\n## F. Controlled Tuning Results (Top 10 Configs)\n")
    wmd("| Rank | leaves | depth | lr | mcs | reg_alpha | reg_lambda | spw | n_trees | Train PR | Val PR | Gap | Prec@1% | Prec@5% |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for rank, r in enumerate(tune_results[:10], 1):
        wmd(f"| {rank} | {r['nl']} | {r['md']} | {r['lr']} | {r['mcs']} | {r['ra']} | {r['rl']} | {r['spw']} | {r['n_best']} | {r['pr_t']:.5f} | {r['pr_v']:.5f} | {r['gap']:+.5f} | {r['p1']*100:.2f}% | {r['p5']*100:.2f}% |")

    # ---- Step 4: Best Model Full Metrics ----
    best_probs = best_tune['probs']
    print(f"\nBest tuned PR-AUC: {best_tune['pr_v']:.5f}")

    # Re-build best model for roc
    best_params_full = next(p for p in grid if
        p['num_leaves'] == best_tune['nl'] and p['max_depth'] == best_tune['md'] and
        p['learning_rate'] == best_tune['lr'] and p['min_child_samples'] == best_tune['mcs'] and
        p['reg_lambda'] == best_tune['rl'] and round(p['scale_pos_weight'],1) == best_tune['spw'])
    best_params_full['n_estimators'] = best_tune['n_best']
    del best_params_full['n_estimators']
    mdl_best = LGBMClassifier(n_estimators=best_tune['n_best'],
                               num_leaves=best_tune['nl'], max_depth=best_tune['md'],
                               learning_rate=best_tune['lr'], min_child_samples=best_tune['mcs'],
                               reg_lambda=best_tune['rl'], scale_pos_weight=best_tune['spw'],
                               subsample=0.9, colsample_bytree=0.9, reg_alpha=0.1,
                               random_state=42, n_jobs=-1)
    mdl_best.fit(X_tr_pre, y_tr)
    roc_best = roc_auc_score(y_va, best_probs)
    p001_b, r001_b, _, _ = get_topk(best_probs, ya, 0.001)
    p1_b,   r1_b,   _, _ = get_topk(best_probs, ya, 0.01)
    p5_b,   r5_b,   _, _ = get_topk(best_probs, ya, 0.05)

    wmd("\n## G. Best Model Metrics\n")
    wmd(f"- **Val PR-AUC**: {best_tune['pr_v']:.5f}")
    wmd(f"- **Train PR-AUC**: {best_tune['pr_t']:.5f}")
    wmd(f"- **Gap**: {best_tune['gap']:+.5f}")
    wmd(f"- **ROC-AUC**: {roc_best:.5f}")
    wmd(f"- **Precision@0.1%**: {p001_b*100:.2f}%")
    wmd(f"- **Precision@1%**: {p1_b*100:.2f}%")
    wmd(f"- **Precision@5%**: {p5_b*100:.2f}%")
    wmd(f"- **Recall@1%**: {r1_b*100:.2f}%")
    wmd(f"- **Recall@5%**: {r5_b*100:.2f}%")
    wmd(f"- **Best n_estimators** (early stopping): {best_tune['n_best']}")

    # ---- Step 5: Threshold Analysis ----
    thresholds = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50]
    # Add best F1 threshold
    precisions, recalls, thresh = precision_recall_curve(y_va, best_probs)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    best_f1_thresh = thresh[np.argmax(f1s[:-1])]
    thresholds.append(float(best_f1_thresh))
    thresholds = sorted(thresholds)

    n_val = len(ya)
    total_fraud = ya.sum()

    wmd("\n## H. Threshold Analysis (Best Tuned Model)\n")
    wmd("| Threshold | Pred Fraud | Pred Rate | TP | FP | FN | TN | Precision | Recall | F1 | FPR | FNR |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
    for t in thresholds:
        pred = (best_probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(ya, pred).ravel() if pred.sum() > 0 else (n_val - total_fraud, 0, total_fraud, 0)
        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn, 1)
        f1v  = 2*prec*rec / max(prec + rec, 1e-9)
        fpr  = fp / max(fp + tn, 1)
        fnr  = fn / max(fn + tp, 1)
        pred_fraud = tp + fp
        wmd(f"| {t:.4f} | {pred_fraud:,} | {pred_fraud/n_val*100:.2f}% | {tp:,} | {fp:,} | {fn:,} | {tn:,} | {prec*100:.2f}% | {rec*100:.2f}% | {f1v:.4f} | {fpr*100:.3f}% | {fnr*100:.2f}% |")

    # ---- Top-K Analysis ----
    wmd("\n## I. Top-K Analysis\n")
    wmd("| K% | K Count | Fraud Captured | Precision@K | Recall@K | Lift@K |\n|---|---|---|---|---|---|")
    for pct in [0.001, 0.005, 0.01, 0.02, 0.05, 0.10]:
        k = max(1, int(n_val * pct))
        idx = np.argsort(best_probs)[::-1][:k]
        fc = ya[idx].sum()
        prec_k = fc / k
        rec_k = fc / max(total_fraud, 1)
        lift_k = prec_k / vfr
        wmd(f"| {pct*100:.1f}% | {k:,} | {fc:,} | {prec_k*100:.2f}% | {rec_k*100:.2f}% | {lift_k:.2f}x |")

    # ---- Time Stability ----
    wmd("\n## J. Time Stability Review\n")
    blocks = block_stats(best_probs, ya)
    wmd("| Block | Fraud Rate | PR-AUC | Prec@1% | Prec@5% | Rec@1% |\n|---|---|---|---|---|---|")
    for b in blocks:
        pr_str = f"{b['pr']:.5f}" if b['pr'] else "N/A"
        wmd(f"| {b['block']} | {b['fr']*100:.4f}% | {pr_str} | {(b['p1'] or 0)*100:.2f}% | {(b['p5'] or 0)*100:.2f}% | {(b['r1'] or 0)*100:.2f}% |")

    # Stability check
    pr_vals = [b['pr'] for b in blocks if b['pr']]
    if pr_vals:
        pr_range = max(pr_vals) - min(pr_vals)
        wmd(f"\nPR-AUC range across blocks: {pr_range:.5f} ({'Stable ✅' if pr_range < 0.005 else 'Unstable ⚠️'})")

    # ---- Sanity Checks ----
    print("\n=== Sanity Checks ===")

    # Label permutation
    y_tr_shuf = y_tr.sample(frac=1, random_state=42).reset_index(drop=True)
    mdl_shuf = LGBMClassifier(n_estimators=best_tune['n_best'],
                               num_leaves=best_tune['nl'], max_depth=best_tune['md'],
                               learning_rate=best_tune['lr'], min_child_samples=best_tune['mcs'],
                               reg_lambda=best_tune['rl'], scale_pos_weight=best_tune['spw'],
                               subsample=0.9, colsample_bytree=0.9, reg_alpha=0.1,
                               random_state=42, n_jobs=-1)
    mdl_shuf.fit(X_tr_pre, y_tr_shuf)
    p_shuf = mdl_shuf.predict_proba(X_va_pre)[:, 1]
    pr_shuf = average_precision_score(y_va, p_shuf)
    gap_perm = (best_tune['pr_v'] - pr_shuf) / pr_shuf * 100
    print(f"  Real PR: {best_tune['pr_v']:.5f}  Shuffle PR: {pr_shuf:.5f}  Gap: {gap_perm:+.1f}%")

    # With/without TSLT
    no_tslt_nums = vc([f for f in tune_nums if f not in TSLT_FEATS])
    pre_notslt = build_preprocessor(no_tslt_nums)
    X_tr_nt = pre_notslt.fit_transform(train[no_tslt_nums + CAT_COLS])
    X_va_nt = pre_notslt.transform(val[no_tslt_nums + CAT_COLS])
    mdl_nt = LGBMClassifier(n_estimators=best_tune['n_best'],
                              num_leaves=best_tune['nl'], max_depth=best_tune['md'],
                              learning_rate=best_tune['lr'], min_child_samples=best_tune['mcs'],
                              reg_lambda=best_tune['rl'], scale_pos_weight=best_tune['spw'],
                              subsample=0.9, colsample_bytree=0.9, reg_alpha=0.1,
                              random_state=42, n_jobs=-1)
    mdl_nt.fit(X_tr_nt, y_tr)
    p_nt = mdl_nt.predict_proba(X_va_nt)[:, 1]
    pr_nt = average_precision_score(y_va, p_nt)

    # Hard subset
    tr_h = train[train['tslt_is_missing'] == 0]
    va_h = val[val['tslt_is_missing'] == 0]
    hard_nums = vc([f for f in tune_nums if f != 'tslt_is_missing'])
    pre_h = build_preprocessor(hard_nums)
    X_tr_h = pre_h.fit_transform(tr_h[hard_nums + CAT_COLS])
    X_va_h = pre_h.transform(va_h[hard_nums + CAT_COLS])
    y_tr_h = tr_h['is_fraud'].astype(int)
    y_va_h = va_h['is_fraud'].astype(int)
    spw_h = (len(y_tr_h) - y_tr_h.sum()) / max(y_tr_h.sum(), 1)
    mdl_h = LGBMClassifier(n_estimators=best_tune['n_best'],
                             num_leaves=best_tune['nl'], max_depth=best_tune['md'],
                             learning_rate=best_tune['lr'], min_child_samples=best_tune['mcs'],
                             reg_lambda=best_tune['rl'], scale_pos_weight=spw_h,
                             subsample=0.9, colsample_bytree=0.9, reg_alpha=0.1,
                             random_state=42, n_jobs=-1)
    mdl_h.fit(X_tr_h, y_tr_h)
    p_h = mdl_h.predict_proba(X_va_h)[:, 1]
    pr_h = average_precision_score(y_va_h, p_h)
    rnd_h = y_va_h.mean()

    wmd("\n## K. Sanity Checks\n")
    wmd("### Label Permutation Test\n")
    wmd(f"| | PR-AUC | Rel Lift | Gap vs Shuffle |\n|---|---|---|---|")
    wmd(f"| Real Labels | {best_tune['pr_v']:.5f} | {best_tune['pr_v']/vfr:.3f}x | — |")
    wmd(f"| Shuffled Labels | {pr_shuf:.5f} | {pr_shuf/vfr:.3f}x | — |")
    wmd(f"| **Difference** | **{best_tune['pr_v']-pr_shuf:+.5f}** | | **{gap_perm:+.1f}%** |")

    if gap_perm > 20:
        perm_verdict = "🟢 **REAL SIGNAL**: Real labels significantly outperform shuffled. Improvement is genuine."
    elif gap_perm > 5:
        perm_verdict = "🟡 **WEAK SIGNAL**: Marginal improvement over shuffle. Treat with caution."
    else:
        perm_verdict = "🔴 **NO SIGNAL**: Model performs similarly with shuffled labels. Improvement may be spurious."
    wmd(f"\n{perm_verdict}\n")

    wmd("### With / Without TSLT\n")
    wmd(f"| Model | Val PR-AUC | Rel Lift |\n|---|---|---|")
    wmd(f"| With TSLT (best tuned) | {best_tune['pr_v']:.5f} | {best_tune['pr_v']/vfr:.3f}x |")
    wmd(f"| Without TSLT | {pr_nt:.5f} | {pr_nt/vfr:.3f}x |")
    wmd(f"| TSLT contribution | {best_tune['pr_v']-pr_nt:+.5f} | |")

    wmd("\n### Hard Subset (tslt_is_missing == 0)\n")
    wmd(f"| | PR-AUC | Random Baseline | Rel Lift |\n|---|---|---|---|")
    wmd(f"| Hard Subset Model | {pr_h:.5f} | {rnd_h:.5f} | {pr_h/rnd_h:.3f}x |")

    # ---- Final Recommendation ----
    best_pr = best_tune['pr_v']
    if best_pr < 0.050:
        sig = "**WEAK** (PR-AUC < 0.05)"
    elif best_pr < 0.070:
        sig = "**WEAK-TO-MODERATE** (PR-AUC 0.05–0.07)"
    elif best_pr < 0.100:
        sig = "**MODERATE** (PR-AUC 0.07–0.10)"
    else:
        sig = "**STRONG** (PR-AUC > 0.10)"

    wmd(f"""
## L. Final Recommendation

### Summary
- **Random Baseline PR-AUC**: {vfr:.5f}
- **Previous Baseline PR-AUC (FS0 baseline)**: {fs0_pr:.5f}
- **Best Feature Set**: {best_fs_name} → Val PR={best_fs_pr:.5f}
- **Best Tuned Model Val PR-AUC**: {best_pr:.5f}
- **Improvement vs FS0**: {(best_pr - fs0_pr):+.5f}
- **Improvement vs random**: {(best_pr / vfr):.3f}x

### Answers
1. **Feature set tốt nhất**: {best_fs_name}
2. **Model tốt nhất**: LightGBM num_leaves={best_tune['nl']}, depth={best_tune['md']}, lr={best_tune['lr']}, mcs={best_tune['mcs']}, reg_lambda={best_tune['rl']}, spw={best_tune['spw']}, n_trees={best_tune['n_best']}
3. **Improvement có thật không?** {"Yes — label permutation confirms gap of " + f"{gap_perm:.1f}%" if gap_perm > 10 else "Marginal — permutation gap " + f"{gap_perm:.1f}%."}
4. **Signal level**: {sig}
5. **Có overfit không?** {"Yes — train/val gap > 0.015, monitor carefully." if best_tune['gap'] > 0.015 else "Minimal — train/val gap is reasonable."}
6. **Có nên tiếp tục feature engineering không?** {"No — current features are near the ceiling for this dataset." if best_pr > 0.048 else "Yes — more signal discovery needed."}
7. **Có nên tune thêm không?** {"Consider deeper tuning if test performance confirmed." if best_pr > 0.048 else "Not yet — signal too weak."}
8. **Có được mở test.csv chưa?** **NO — test.csv remains locked. Pipeline must be frozen first.**
""")

    print(f"\n✅ Stabilization & Tuning Completed!")
    print(f"Best Val PR-AUC: {best_pr:.5f} | Permutation Gap: {gap_perm:+.1f}%")

if __name__ == "__main__":
    main()
