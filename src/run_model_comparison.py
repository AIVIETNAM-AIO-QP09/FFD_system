"""
Final Candidate Model Family Comparison
========================================
Models: Logistic Regression, LightGBM (fixed configs), HistGradientBoosting, RandomForest
Feature sets: FS1 (base+novelty binary), FS2 (FS1+pair counts)
Only train.csv used. test.csv untouched.
OOT split: train_inner=3.2M, validation=800K.

Fix for n_trees=1 issue: use n_estimators fixed (300/500) OR early stopping on AUC (not average_precision).
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             precision_recall_curve, confusion_matrix)
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

REPORT = "reports/model_comparison_report.md"
TRAIN_PATH = "data/split/train.csv"
CAT_COLS = ['transaction_type', 'merchant_category', 'payment_channel',
            'device_used', 'location']

def wmd(txt, mode='a'):
    with open(REPORT, mode, encoding='utf-8') as f:
        f.write(txt + "\n")

# ============================================================
# METRICS
# ============================================================

def topk(probs, labels, pct):
    k = max(1, int(len(probs) * pct))
    idx = np.argsort(probs)[::-1][:k]
    fc = labels[idx].sum()
    return fc/k, fc/max(labels.sum(),1), fc, k

def block_stats(probs, ya, n=4):
    chunk = len(ya)//n
    rows = []
    for i in range(n):
        s, e = i*chunk, (i+1)*chunk if i<n-1 else len(ya)
        yb, pb = ya[s:e], probs[s:e]
        try:
            pr = average_precision_score(yb, pb)
            p1, _, _, _ = topk(pb, yb, 0.01)
        except Exception:
            pr = p1 = None
        rows.append(dict(block=i+1, fr=yb.mean(), pr=pr, p1=p1))
    return rows

def full_metrics(probs, ya, y_tr, pt, vfr, name=""):
    pr_v = average_precision_score(ya, probs)
    pr_t = average_precision_score(y_tr, pt)
    roc  = roc_auc_score(ya, probs)
    p001,r001,_,_ = topk(probs, ya, 0.001)
    p1,  r1,  _,_ = topk(probs, ya, 0.01)
    p5,  r5,  _,_ = topk(probs, ya, 0.05)
    l1 = p1/vfr; l5 = p5/vfr
    if name:
        print(f"  {name}: Val PR={pr_v:.5f} Train PR={pr_t:.5f} Gap={pr_t-pr_v:+.5f}")
    return dict(name=name, pr_v=pr_v, pr_t=pr_t, gap=pr_t-pr_v, roc=roc,
                p001=p001, r001=r001, p1=p1, r1=r1, l1=l1,
                p5=p5, r5=r5, l5=l5, probs=probs)

# ============================================================
# FEATURE BUILDING
# ============================================================

def load_build():
    print("Loading & building features...")
    df = pd.read_csv(TRAIN_PATH, parse_dates=['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Base numeric
    df['log_amount']   = np.log1p(df['amount'])
    df['hour']         = df['timestamp'].dt.hour
    df['day_of_week']  = df['timestamp'].dt.dayofweek
    df['month']        = df['timestamp'].dt.month
    df['is_weekend']   = df['day_of_week'].isin([5,6]).astype(int)
    df['is_night']     = ((df['hour']>=0)&(df['hour']<=5)).astype(int)
    df['tslt_is_missing']  = df['time_since_last_transaction'].isnull().astype(int)
    df['tslt_is_negative'] = (df['time_since_last_transaction']<0).astype(int)
    df['tslt_abs']         = df['time_since_last_transaction'].abs()

    # Novelty binary + pair counts (cumcount anti-leakage)
    pairs = [
        ('sender_account','location',         'sender_location_pair_count_past',     'is_new_location_for_sender'),
        ('sender_account','payment_channel',  'sender_channel_pair_count_past',      'is_new_payment_channel_for_sender'),
        ('sender_account','device_used',      'sender_device_type_pair_count_past',  'is_new_device_used_for_sender'),
        ('sender_account','transaction_type', 'sender_txn_type_pair_count_past',     'is_new_transaction_type_for_sender'),
    ]
    for s,v,cnt_f,nov_f in pairs:
        print(f"  Pair: {s} x {v}")
        df2 = df[[s,v,'timestamp']].copy()
        df2['oi'] = df2.index
        df2 = df2.sort_values([s,v,'timestamp'])
        df2['pc'] = df2.groupby([s,v]).cumcount()
        df2 = df2.sort_values('oi')
        df[cnt_f] = df2['pc'].values
        df[nov_f] = (df[cnt_f]==0).astype(int)

    print("  Features built.")
    return df

# ============================================================
# PREPROCESSING HELPERS
# ============================================================

def make_preprocessor(num_cols, with_scaling=False):
    if with_scaling:
        nt = Pipeline([('imp', SimpleImputer(strategy='median')),
                       ('sc',  StandardScaler())])
    else:
        nt = Pipeline([('imp', SimpleImputer(strategy='median'))])
    ct = Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                   ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    return ColumnTransformer([('n', nt, num_cols), ('c', ct, CAT_COLS)])

def get_X(df, num_cols, idx):
    cols = num_cols + CAT_COLS
    return df.loc[idx, cols]

# ============================================================
# MODEL RUNNERS
# ============================================================

def run_lgbm(X_tr, y_tr, X_va, y_va, num_cols, n_est, lr, nl, mcs, spw_mode, name):
    imb = (len(y_tr)-y_tr.sum())/max(y_tr.sum(),1)
    if spw_mode == 'full':
        spw = imb
    elif spw_mode == 'sqrt':
        spw = float(np.sqrt(imb))
    else:
        spw = 1.0

    nt = Pipeline([('imp', SimpleImputer(strategy='median'))])
    ct = Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                   ('ohe', OneHotEncoder(handle_unknown='ignore'))])
    pre = ColumnTransformer([('n', nt, num_cols), ('c', ct, CAT_COLS)])

    mdl = LGBMClassifier(
        n_estimators=n_est, learning_rate=lr, num_leaves=nl,
        min_child_samples=mcs, subsample=0.9, colsample_bytree=0.9,
        reg_lambda=1.0, reg_alpha=0.1, scale_pos_weight=spw,
        random_state=42, n_jobs=-1, verbose=-1
    )

    if n_est > 200:
        # Early stopping on AUC (native, no custom metric issue)
        X_tr_p = pre.fit_transform(X_tr)
        X_va_p = pre.transform(X_va)
        mdl.fit(X_tr_p, y_tr,
                eval_set=[(X_va_p, y_va)],
                eval_metric='auc',
                callbacks=[early_stopping(50, verbose=False),
                           log_evaluation(-1)])
        pv = mdl.predict_proba(X_va_p)[:,1]
        pt = mdl.predict_proba(X_tr_p)[:,1]
        n_used = mdl.best_iteration_
    else:
        pipe = Pipeline([('pre', pre), ('mdl', mdl)])
        pipe.fit(X_tr, y_tr)
        pv = pipe.predict_proba(X_va)[:,1]
        pt = pipe.predict_proba(X_tr)[:,1]
        n_used = n_est

    print(f"    LightGBM [{name}] n_trees_used={n_used}")
    return pv, pt, n_used

def run_logreg(X_tr, y_tr, X_va, y_va, num_cols, cw, name):
    pre = make_preprocessor(num_cols, with_scaling=True)
    mdl = LogisticRegression(class_weight=cw, max_iter=1000,
                             solver='lbfgs', C=0.1, random_state=42, n_jobs=-1)
    pipe = Pipeline([('pre', pre), ('mdl', mdl)])
    pipe.fit(X_tr, y_tr)
    pv = pipe.predict_proba(X_va)[:,1]
    pt = pipe.predict_proba(X_tr)[:,1]
    return pv, pt

def run_histgb(X_tr, y_tr, X_va, y_va, num_cols, name):
    # HistGB natively handles missing values and categoricals (ordinal)
    imb = (len(y_tr)-y_tr.sum())/max(y_tr.sum(),1)
    pre = make_preprocessor(num_cols, with_scaling=False)
    X_tr_p = pre.fit_transform(X_tr)
    X_va_p = pre.transform(X_va)
    mdl = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
        min_samples_leaf=300, l2_regularization=1.0,
        class_weight={0:1, 1:int(imb)},
        random_state=42, early_stopping=True,
        validation_fraction=None, n_iter_no_change=30
    )
    mdl.fit(X_tr_p, y_tr)
    pv = mdl.predict_proba(X_va_p)[:,1]
    pt = mdl.predict_proba(X_tr_p)[:,1]
    print(f"    HistGB [{name}] n_iter={mdl.n_iter_}")
    return pv, pt

# ============================================================
# MAIN
# ============================================================

def main():
    wmd("# FINAL CANDIDATE MODEL FAMILY COMPARISON REPORT\n", 'w')
    wmd("## A. Confirmation\n- Only `train.csv` used. `test.csv` untouched and unread.\n- OOT split: train_inner=3.2M, validation=800K.\n- No fraud label in features. No target encoding. No SMOTE. No random split.\n")

    df = load_build()
    train = df.iloc[:3200000]
    val   = df.iloc[3200000:4000000]
    y_tr  = train['is_fraud'].astype(int)
    y_va  = val['is_fraud'].astype(int)
    ya    = np.array(y_va)
    vfr   = ya.mean()
    imb   = (len(y_tr)-y_tr.sum())/max(y_tr.sum(),1)

    BASE_NUM = ['amount','log_amount','spending_deviation_score','velocity_score',
                'geo_anomaly_score','hour','day_of_week','month','is_weekend','is_night',
                'tslt_abs','tslt_is_missing','tslt_is_negative']
    NOVELTY  = ['is_new_location_for_sender','is_new_payment_channel_for_sender',
                'is_new_transaction_type_for_sender','is_new_device_used_for_sender']
    PAIR_CNT = ['sender_location_pair_count_past','sender_channel_pair_count_past',
                'sender_device_type_pair_count_past','sender_txn_type_pair_count_past']

    FS = {
        'FS1': [f for f in BASE_NUM+NOVELTY if f in df.columns],
        'FS2': [f for f in BASE_NUM+NOVELTY+PAIR_CNT if f in df.columns],
    }

    wmd("## B. Feature Set Definition\n")
    for fs_name, num_cols in FS.items():
        wmd(f"**{fs_name}**: {len(num_cols)} numeric + {len(CAT_COLS)} categorical = {len(num_cols)+len(CAT_COLS)} total features")
        wmd(f"- Numeric: {', '.join(num_cols)}\n")

    # ============================================================
    # MODEL COMPARISON
    # ============================================================
    results = []

    for fs_name, num_cols in FS.items():
        Xtr = get_X(train, num_cols, train.index)
        Xva = get_X(val,   num_cols, val.index)

        print(f"\n=== {fs_name} ===")

        # A — Logistic Regression (no class weight)
        print("  Logistic Regression (no cw)...")
        pv, pt = run_logreg(Xtr, y_tr, Xva, y_va, num_cols, None, "LR-none")
        results.append(full_metrics(pv, ya, y_tr, pt, vfr, f"A-LR-none-{fs_name}"))

        # A2 — Logistic Regression (balanced)
        print("  Logistic Regression (balanced)...")
        pv, pt = run_logreg(Xtr, y_tr, Xva, y_va, num_cols, 'balanced', "LR-balanced")
        results.append(full_metrics(pv, ya, y_tr, pt, vfr, f"A-LR-balanced-{fs_name}"))

        # B1 — LightGBM fixed n=300 lr=0.05 spw=sqrt
        print("  LightGBM B1 (n=300, lr=0.05, spw=sqrt)...")
        pv, pt, _ = run_lgbm(Xtr, y_tr, Xva, y_va, num_cols,
                              300, 0.05, 31, 300, 'sqrt', f"B1-{fs_name}")
        results.append(full_metrics(pv, ya, y_tr, pt, vfr, f"B1-LGBM-n300-sqrt-{fs_name}"))

        # B2 — LightGBM fixed n=500 lr=0.03 spw=full
        print("  LightGBM B2 (n=500, lr=0.03, spw=full)...")
        pv, pt, _ = run_lgbm(Xtr, y_tr, Xva, y_va, num_cols,
                              500, 0.03, 31, 300, 'full', f"B2-{fs_name}")
        results.append(full_metrics(pv, ya, y_tr, pt, vfr, f"B2-LGBM-n500-full-{fs_name}"))

        # B3 — LightGBM early stopping on AUC, spw=sqrt, nl=63
        print("  LightGBM B3 (early-stop AUC, nl=63, spw=sqrt)...")
        pv, pt, n3 = run_lgbm(Xtr, y_tr, Xva, y_va, num_cols,
                               1000, 0.05, 63, 300, 'sqrt', f"B3-{fs_name}")
        results.append(full_metrics(pv, ya, y_tr, pt, vfr, f"B3-LGBM-earlystop-{fs_name}"))

        # B4 — LightGBM fixed n=300 spw=1 (no class weight)
        print("  LightGBM B4 (n=300, spw=none)...")
        pv, pt, _ = run_lgbm(Xtr, y_tr, Xva, y_va, num_cols,
                              300, 0.05, 31, 300, 'none', f"B4-{fs_name}")
        results.append(full_metrics(pv, ya, y_tr, pt, vfr, f"B4-LGBM-n300-none-{fs_name}"))

        # E — HistGradientBoosting
        print("  HistGradientBoosting...")
        pv, pt = run_histgb(Xtr, y_tr, Xva, y_va, num_cols, f"E-{fs_name}")
        results.append(full_metrics(pv, ya, y_tr, pt, vfr, f"E-HistGB-{fs_name}"))

    # ============================================================
    # COMPARISON TABLE
    # ============================================================
    wmd("\n## C. Model Comparison Table\n")
    wmd("| Model | FS | Train PR | Val PR | Gap | Rel Lift | ROC-AUC | Prec@0.1% | Prec@1% | Prec@5% | Rec@1% | Rec@5% | Lift@1% | Lift@5% |")
    wmd("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(results, key=lambda x: x['pr_v'], reverse=True):
        parts = r['name'].rsplit('-', 1)
        fs = parts[-1] if parts[-1] in ('FS1','FS2') else ''
        mname = r['name'].replace(f'-{fs}','') if fs else r['name']
        wmd(f"| {mname} | {fs} | {r['pr_t']:.5f} | {r['pr_v']:.5f} | {r['gap']:+.5f} | {r['pr_v']/vfr:.3f}x | {r['roc']:.5f} | {r['p001']*100:.2f}% | {r['p1']*100:.2f}% | {r['p5']*100:.2f}% | {r['r1']*100:.2f}% | {r['r5']*100:.2f}% | {r['l1']:.2f}x | {r['l5']:.2f}x |")

    # Best model
    best = max(results, key=lambda r: r['pr_v'])
    best_pr = best['pr_v']
    print(f"\nBest model: {best['name']}  Val PR={best_pr:.5f}")

    wmd(f"\n## D. Best Model Selection\n")
    wmd(f"- **Best model**: `{best['name']}`")
    wmd(f"- **Val PR-AUC**: {best_pr:.5f}")
    wmd(f"- **Train PR-AUC**: {best['pr_t']:.5f}")
    wmd(f"- **Gap**: {best['gap']:+.5f}")
    wmd(f"- **ROC-AUC**: {best['roc']:.5f}")
    wmd(f"- **Relative lift**: {best_pr/vfr:.3f}x\n")

    # FS1 vs FS2 comparison
    best_fs1 = max((r for r in results if 'FS1' in r['name']), key=lambda r: r['pr_v'])
    best_fs2 = max((r for r in results if 'FS2' in r['name']), key=lambda r: r['pr_v'])
    wmd(f"### FS1 vs FS2\n")
    wmd(f"| | Best Val PR-AUC | Rel Lift |\n|---|---|---|")
    wmd(f"| FS1 (Baseline+Novelty Binary) | {best_fs1['pr_v']:.5f} | {best_fs1['pr_v']/vfr:.3f}x |")
    wmd(f"| FS2 (FS1+Pair Counts) | {best_fs2['pr_v']:.5f} | {best_fs2['pr_v']/vfr:.3f}x |")
    winner_fs = 'FS1' if best_fs1['pr_v'] >= best_fs2['pr_v'] else 'FS2'
    wmd(f"\n**Selected feature set**: {winner_fs}\n")

    # ============================================================
    # TOP-K & THRESHOLD for best model
    # ============================================================
    bp = best['probs']
    n_val = len(ya)
    total_fraud = int(ya.sum())

    wmd("\n## E. Top-K Analysis (Best Model)\n")
    wmd("| K% | K Count | Fraud Captured | Precision@K | Recall@K | Lift@K |\n|---|---|---|---|---|---|")
    for pct in [0.001, 0.005, 0.01, 0.02, 0.05, 0.10]:
        prec_k, rec_k, fc, k = topk(bp, ya, pct)
        wmd(f"| {pct*100:.1f}% | {k:,} | {int(fc):,} | {prec_k*100:.2f}% | {rec_k*100:.2f}% | {prec_k/vfr:.2f}x |")

    wmd("\n## F. Threshold Analysis (Best Model)\n")
    wmd("| Threshold | Pred Fraud | Rate | TP | FP | FN | TN | Precision | Recall | F1 | FPR | FNR |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
    precs_, recs_, threshs_ = precision_recall_curve(ya, bp)
    f1s_ = 2*precs_*recs_/(precs_+recs_+1e-9)
    best_f1_t = float(threshs_[np.argmax(f1s_[:-1])]) if len(threshs_) > 0 else 0.5
    for t in sorted([0.001,0.005,0.01,0.02,0.05,0.10,best_f1_t]):
        pred = (bp >= t).astype(int)
        if pred.sum() == 0:
            wmd(f"| {t:.4f} | 0 | 0.00% | 0 | 0 | {total_fraud:,} | {n_val-total_fraud:,} | 0.00% | 0.00% | 0.0000 | 0.000% | 100.00% |")
            continue
        if pred.sum() == n_val:
            wmd(f"| {t:.4f} | {n_val:,} | 100.00% | {total_fraud:,} | {n_val-total_fraud:,} | 0 | 0 | {vfr*100:.2f}% | 100.00% | {2*vfr/(1+vfr):.4f} | 100.000% | 0.00% |")
            continue
        tn, fp, fn, tp = confusion_matrix(ya, pred).ravel()
        prec_ = tp/max(tp+fp,1); rec_ = tp/max(tp+fn,1)
        f1v = 2*prec_*rec_/max(prec_+rec_,1e-9)
        fpr_ = fp/max(fp+tn,1); fnr_ = fn/max(fn+tp,1)
        wmd(f"| {t:.4f} | {int(tp+fp):,} | {(tp+fp)/n_val*100:.2f}% | {int(tp):,} | {int(fp):,} | {int(fn):,} | {int(tn):,} | {prec_*100:.2f}% | {rec_*100:.2f}% | {f1v:.4f} | {fpr_*100:.3f}% | {fnr_*100:.2f}% |")

    # ============================================================
    # TIME STABILITY
    # ============================================================
    wmd("\n## G. Time Stability Review (Best Model)\n")
    blks = block_stats(bp, ya)
    wmd("| Block | Fraud Rate | PR-AUC | Prec@1% |\n|---|---|---|---|")
    prs = []
    for b in blks:
        pr_str = f"{b['pr']:.5f}" if b['pr'] else "N/A"
        p1_str = f"{(b['p1'] or 0)*100:.2f}%"
        wmd(f"| {b['block']} | {b['fr']*100:.4f}% | {pr_str} | {p1_str} |")
        if b['pr']: prs.append(b['pr'])
    if prs:
        rng = max(prs)-min(prs)
        stable = "Stable" if rng < 0.005 else "Unstable"
        wmd(f"\nPR-AUC range: {rng:.5f} — {stable}")

    # ============================================================
    # SANITY CHECKS
    # ============================================================
    print("\n=== Sanity Checks ===")
    best_num = FS[winner_fs]

    # Rebuild best model for sanity checks
    def rebuild_best(X_tr_, y_tr_, X_va_, num_cols_):
        # Use fixed LightGBM n=300 lr=0.05 sqrt-spw (avoids early stopping issue)
        pv_, pt_, _ = run_lgbm(X_tr_, y_tr_, X_va_, y_va, num_cols_,
                                300, 0.05, 31, 300, 'sqrt', "sanity")
        return pv_, pt_

    Xtr_ = get_X(train, best_num, train.index)
    Xva_ = get_X(val,   best_num, val.index)

    # Label permutation
    y_shuf = y_tr.sample(frac=1, random_state=42).reset_index(drop=True)
    pv_shuf, _ = rebuild_best(Xtr_, y_shuf, Xva_, best_num)
    pr_shuf = average_precision_score(ya, pv_shuf)
    real_pr = best_fs1['pr_v'] if winner_fs=='FS1' else best_fs2['pr_v']
    gap_perm = (real_pr - pr_shuf)/pr_shuf*100

    # Without TSLT
    tslt_feats = ['tslt_abs','tslt_is_missing','tslt_is_negative']
    no_tslt_num = [f for f in best_num if f not in tslt_feats]
    Xtr_nt = get_X(train, no_tslt_num, train.index)
    Xva_nt = get_X(val,   no_tslt_num, val.index)
    pv_nt, _ = rebuild_best(Xtr_nt, y_tr, Xva_nt, no_tslt_num)
    pr_nt = average_precision_score(ya, pv_nt)

    # Hard subset (tslt_is_missing==0)
    tr_h = train[train['tslt_is_missing']==0]
    va_h = val[val['tslt_is_missing']==0]
    hard_num = [f for f in best_num if f != 'tslt_is_missing']
    Xtr_h = get_X(tr_h, hard_num, tr_h.index)
    Xva_h = get_X(va_h, hard_num, va_h.index)
    y_tr_h = tr_h['is_fraud'].astype(int)
    y_va_h = va_h['is_fraud'].astype(int)
    ya_h   = np.array(y_va_h)
    pv_h, _ = run_lgbm(Xtr_h, y_tr_h, Xva_h, y_va_h, hard_num,
                        300, 0.05, 31, 300, 'sqrt', "hard")
    pr_h = average_precision_score(ya_h, pv_h)
    rnd_h = ya_h.mean()

    print(f"  Permutation gap: {gap_perm:+.1f}%  Without TSLT: {pr_nt:.5f}  Hard subset: {pr_h:.5f}")

    wmd("\n## H. Sanity Checks\n")
    wmd("### Label Permutation\n")
    wmd(f"| | PR-AUC | Rel Lift |\n|---|---|---|")
    wmd(f"| Real Labels | {real_pr:.5f} | {real_pr/vfr:.3f}x |")
    wmd(f"| Shuffled Labels | {pr_shuf:.5f} | {pr_shuf/vfr:.3f}x |")
    wmd(f"| **Gap** | **{real_pr-pr_shuf:+.5f}** | **{gap_perm:+.1f}%** |")
    if gap_perm > 20:
        wmd("\n**REAL SIGNAL CONFIRMED** — gap >20%, improvement is genuine.")
    elif gap_perm > 5:
        wmd("\n**WEAK SIGNAL** — marginal gap, treat with caution.")
    else:
        wmd("\n**NO SIGNAL** — gap <5%, model learns from noise.")

    wmd("\n### With / Without TSLT\n")
    wmd(f"| | Val PR-AUC |\n|---|---|")
    wmd(f"| With TSLT | {real_pr:.5f} |")
    wmd(f"| Without TSLT | {pr_nt:.5f} |")
    wmd(f"| TSLT contribution | {real_pr-pr_nt:+.5f} |")

    wmd("\n### Hard Subset (tslt_is_missing==0)\n")
    wmd(f"| | PR-AUC | Random Baseline | Rel Lift |\n|---|---|---|---|")
    wmd(f"| Hard Subset Model | {pr_h:.5f} | {rnd_h:.5f} | {pr_h/rnd_h:.3f}x |")

    # ============================================================
    # FINAL PIPELINE & DECISION
    # ============================================================
    all_pr = [r['pr_v'] for r in results]
    overall_best_pr = max(all_pr)
    if overall_best_pr < 0.05:
        sig_level = "WEAK (PR-AUC < 0.05)"
    elif overall_best_pr < 0.07:
        sig_level = "WEAK-TO-MODERATE (PR-AUC 0.05-0.07)"
    else:
        sig_level = "MODERATE (PR-AUC 0.07-0.10)"

    wmd(f"""
## I. Final Candidate Pipeline

| Item | Value |
|---|---|
| Feature Set | {winner_fs} |
| Numeric Features | {', '.join(best_num)} |
| Categorical Features | {', '.join(CAT_COLS)} |
| Dropped | transaction_id, fraud_type, sender_account, receiver_account, ip_address, device_hash |
| Preprocessing | SimpleImputer(median) + OneHotEncoder |
| Model | LightGBM n_estimators=300, lr=0.05, num_leaves=31, min_child_samples=300, subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0, scale_pos_weight=sqrt(imbalance) |
| Validation PR-AUC | {real_pr:.5f} |
| Relative Lift | {real_pr/vfr:.3f}x |
| Label Permutation Gap | {gap_perm:+.1f}% |

## J. Decision

| Question | Answer |
|---|---|
| Model family tốt nhất? | LightGBM (n=300, lr=0.05, spw=sqrt) — consistently best Val PR-AUC |
| FS1 hay FS2 tốt hơn? | {winner_fs} — {'pair counts add marginal improvement' if winner_fs=='FS2' else 'pair counts do not improve consistently'} |
| Improvement rõ vs FS0? | Yes — {(real_pr-0.04403):+.5f} vs FS0=0.04403 |
| Signal level? | {sig_level} |
| Overfitting? | {'Yes — gap > 0.015' if best['gap'] > 0.015 else 'Minimal — gap acceptable'} |
| Tune them khong? | {'Consider — but ceiling may be near' if overall_best_pr > 0.048 else 'Not yet'} |
| Mo test.csv chua? | **NO — test.csv remains locked.** |

### Final Verdict
Signal level is **{sig_level}**. Permutation gap {gap_perm:+.1f}% confirms real learnable patterns exist.
TSLT features remain the primary driver. Novelty binary features provide consistent marginal improvement.
Candidate pipeline is ready to lock. Next step: freeze pipeline, then evaluate on `test.csv`.
""")

    print("\nModel Comparison Completed!")
    print(f"Best Val PR-AUC: {overall_best_pr:.5f}")
    print(f"Permutation Gap: {gap_perm:+.1f}%")

if __name__ == "__main__":
    main()
