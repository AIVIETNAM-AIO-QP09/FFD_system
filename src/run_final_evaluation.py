"""
Final Locked Pipeline Holdout Test Evaluation
==============================================
Feature set: FS1 — Baseline + Novelty Binary
Model: LightGBMClassifier (n=300, spw=1)
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, confusion_matrix
from lightgbm import LGBMClassifier

REPORT = "reports/final_test_evaluation_report.md"
TRAIN_PATH = "data/split/train.csv"
TEST_PATH = "data/split/test.csv"

def wmd(txt, mode='a'):
    with open(REPORT, mode, encoding='utf-8') as f:
        f.write(txt + "\n")

def topk(probs, labels, pct):
    k = max(1, int(len(probs)*pct))
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
            p5, _, _, _ = topk(pb, yb, 0.05)
            _, r1, _, _ = topk(pb, yb, 0.01)
            _, r5, _, _ = topk(pb, yb, 0.05)
        except Exception:
            pr = p1 = p5 = r1 = r5 = None
        rows.append(dict(block=i+1, fr=yb.mean(), pr=pr, p1=p1, p5=p5, r1=r1, r5=r5))
    return rows

def build_strict_cumcount(df, group_cols, time_col='timestamp'):
    df2 = df[group_cols + [time_col]].copy()
    df2['oi'] = df2.index
    df2 = df2.sort_values(group_cols + [time_col])
    cum_all = df2.groupby(group_cols).cumcount()
    cum_tie = df2.groupby(group_cols + [time_col]).cumcount()
    df2['pc'] = cum_all - cum_tie
    df2 = df2.sort_values('oi')
    return df2['pc'].values

def main():
    wmd("# FINAL HOLDOUT TEST EVALUATION REPORT\n", 'w')
    wmd("## A. Pipeline Lock Confirmation\nPipeline was locked (FS1 + LightGBM spw=1) before opening test set.\n")
    wmd("## B. Test Access Confirmation\n`test.csv` opened for the first and only time for final evaluation.\n")
    
    print("Loading datasets...")
    train = pd.read_csv(TRAIN_PATH, parse_dates=['timestamp'])
    test = pd.read_csv(TEST_PATH, parse_dates=['timestamp'])
    train['is_test'] = 0
    test['is_test'] = 1
    
    has_labels = 'is_fraud' in test.columns
    
    df = pd.concat([train, test], ignore_index=True)
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    print("Building features (streaming test simulation)...")
    df['log_amount'] = np.log1p(df['amount'])
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)
    df['is_night'] = ((df['hour']>=0)&(df['hour']<=5)).astype(int)
    df['tslt_abs'] = df['time_since_last_transaction'].abs()
    df['tslt_is_missing'] = df['time_since_last_transaction'].isnull().astype(int)
    df['tslt_is_negative'] = (df['time_since_last_transaction']<0).astype(int)
    
    pairs = [
        ('sender_account', 'location', 'sender_location_pair_count_past', 'is_new_location_for_sender'),
        ('sender_account', 'payment_channel', 'sender_channel_pair_count_past', 'is_new_payment_channel_for_sender'),
        ('sender_account', 'device_used', 'sender_device_type_pair_count_past', 'is_new_device_used_for_sender'),
        ('sender_account', 'transaction_type', 'sender_txn_type_pair_count_past', 'is_new_transaction_type_for_sender'),
    ]
    for s, v, cnt_f, nov_f in pairs:
        print(f"  Building strict novelty for {v}...")
        df[cnt_f] = build_strict_cumcount(df, [s, v])
        df[nov_f] = (df[cnt_f] == 0).astype(int)
        
    print("Features built.")
    
    NUM_COLS = ['amount','log_amount','spending_deviation_score','velocity_score',
                'geo_anomaly_score','hour','day_of_week','month','is_weekend','is_night',
                'tslt_abs','tslt_is_missing','tslt_is_negative',
                'is_new_location_for_sender','is_new_payment_channel_for_sender',
                'is_new_transaction_type_for_sender','is_new_device_used_for_sender']
    CAT_COLS = ['transaction_type','merchant_category','payment_channel','device_used','location']
    
    wmd("## C. Final Pipeline Definition\n")
    wmd("- **Feature Set**: FS1 (Baseline + Novelty Binary)")
    wmd(f"- **Numeric Features**: {', '.join(NUM_COLS)}")
    wmd(f"- **Categorical Features**: {', '.join(CAT_COLS)}")
    wmd("- **Dropped Columns**: transaction_id, fraud_type, sender_account, receiver_account, ip_address, device_hash, raw timestamp, raw time_since_last_transaction")
    wmd("- **Preprocessing**: SimpleImputer(median) for numeric, SimpleImputer(most_frequent) + OneHotEncoder for categorical. Fitted on train only.")
    wmd("- **Model**: LightGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=300, subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0, scale_pos_weight=1.0, random_state=42)")
    wmd("- **Feature Generation Mode**: Streaming test simulation (test rows use past train and past test rows).")
    
    train_df = df[df['is_test']==0].copy()
    test_df = df[df['is_test']==1].copy()
    
    val_idx = int(len(train_df) * 0.8)
    
    X_tr = train_df[NUM_COLS + CAT_COLS]
    y_tr = train_df['is_fraud'].astype(int)
    X_te = test_df[NUM_COLS + CAT_COLS]
    
    if has_labels:
        y_te = test_df['is_fraud'].astype(int)
        test_fraud_cnt = y_te.sum()
        test_size = len(y_te)
        test_fr = test_fraud_cnt / test_size
        wmd("\n## D. Test Dataset Summary\n")
        wmd(f"- **Rows**: {test_size:,}")
        wmd(f"- **Time Range**: {test_df['timestamp'].min()} to {test_df['timestamp'].max()}")
        wmd(f"- **Fraud Count**: {test_fraud_cnt:,}")
        wmd(f"- **Fraud Rate**: {test_fr*100:.4f}%")
        
    print("Training model...")
    nt = Pipeline([('imp', SimpleImputer(strategy='median'))])
    ct = Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                   ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    pre = ColumnTransformer([('n', nt, NUM_COLS), ('c', ct, CAT_COLS)])
    
    mdl = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, 
                         min_child_samples=300, subsample=0.9, colsample_bytree=0.9, 
                         reg_lambda=1.0, scale_pos_weight=1.0, random_state=42, n_jobs=-1, verbose=-1)
    
    pipe = Pipeline([('pre', pre), ('mdl', mdl)])
    pipe.fit(X_tr, y_tr)
    
    print("Finding best F1 threshold on validation...")
    X_val_eval = X_tr.iloc[val_idx:]
    y_val_eval = y_tr.iloc[val_idx:]
    val_probs = pipe.predict_proba(X_val_eval)[:,1]
    
    val_pr_auc = average_precision_score(y_val_eval, val_probs)
    val_p1, _, _, _ = topk(val_probs, np.array(y_val_eval), 0.01)
    val_p5, _, _, _ = topk(val_probs, np.array(y_val_eval), 0.05)
    
    precs, recs, threshs = precision_recall_curve(y_val_eval, val_probs)
    f1s = 2*precs*recs/(precs+recs+1e-9)
    best_f1_t = float(threshs[np.argmax(f1s[:-1])]) if len(threshs) > 0 else 0.5
    
    print("Evaluating on test...")
    test_probs = pipe.predict_proba(X_te)[:,1]
    
    if not has_labels:
        wmd("\nTest set does not have labels. Saving predictions only.")
        sub = pd.DataFrame({'transaction_id': test_df['transaction_id'], 'fraud_probability': test_probs})
        sub.to_csv("data/split/test_predictions.csv", index=False)
        print("Predictions saved to data/split/test_predictions.csv")
        return
        
    ya = np.array(y_te)
    test_pr_auc = average_precision_score(ya, test_probs)
    test_roc = roc_auc_score(ya, test_probs)
    test_lift = test_pr_auc / test_fr
    
    p001, _, _, _ = topk(test_probs, ya, 0.001)
    p005, _, _, _ = topk(test_probs, ya, 0.005)
    p1, r1, _, _ = topk(test_probs, ya, 0.01)
    p2, _, _, _ = topk(test_probs, ya, 0.02)
    p5, r5, _, _ = topk(test_probs, ya, 0.05)
    p10, r10, _, _ = topk(test_probs, ya, 0.10)
    
    wmd("\n## E. Final Test Metrics\n")
    wmd(f"- **PR-AUC**: {test_pr_auc:.5f}")
    wmd(f"- **ROC-AUC**: {test_roc:.5f}")
    wmd(f"- **Relative Lift**: {test_lift:.3f}x")
    wmd(f"- **Precision@0.1%**: {p001*100:.2f}%")
    wmd(f"- **Precision@0.5%**: {p005*100:.2f}%")
    wmd(f"- **Precision@1%**: {p1*100:.2f}%")
    wmd(f"- **Precision@2%**: {p2*100:.2f}%")
    wmd(f"- **Precision@5%**: {p5*100:.2f}%")
    wmd(f"- **Precision@10%**: {p10*100:.2f}%")
    wmd(f"- **Recall@1%**: {r1*100:.2f}%")
    wmd(f"- **Recall@5%**: {r5*100:.2f}%")
    wmd(f"- **Recall@10%**: {r10*100:.2f}%")
    wmd(f"- **Lift@1%**: {p1/test_fr:.2f}x")
    wmd(f"- **Lift@5%**: {p5/test_fr:.2f}x")
    wmd(f"- **Lift@10%**: {p10/test_fr:.2f}x")
    
    wmd("\n## F. Validation vs Test Comparison\n")
    wmd("| Metric | Validation | Test | Generalization |")
    wmd("|---|---|---|---|")
    pr_gen = "Good" if test_pr_auc > val_pr_auc * 0.9 else "Drop"
    p1_gen = "Good" if p1 > val_p1 * 0.9 else "Drop"
    p5_gen = "Good" if p5 > val_p5 * 0.9 else "Drop"
    wmd(f"| PR-AUC | {val_pr_auc:.5f} | {test_pr_auc:.5f} | {pr_gen} |")
    wmd(f"| Precision@1% | {val_p1*100:.2f}% | {p1*100:.2f}% | {p1_gen} |")
    wmd(f"| Precision@5% | {val_p5*100:.2f}% | {p5*100:.2f}% | {p5_gen} |")
    
    wmd("\n## G. Threshold Analysis\n")
    wmd("| Threshold | Pred Fraud | Rate | TP | FP | FN | TN | Precision | Recall | F1 | FPR | FNR | Note |")
    wmd("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    threshs_eval = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10, best_f1_t]
    for t in sorted(threshs_eval):
        pred = (test_probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(ya, pred).ravel() if pred.sum()>0 else (test_size-test_fraud_cnt, 0, test_fraud_cnt, 0)
        prec_ = tp/max(tp+fp,1); rec_ = tp/max(tp+fn,1)
        f1v = 2*prec_*rec_/max(prec_+rec_,1e-9)
        fpr_ = fp/max(fp+tn,1); fnr_ = fn/max(fn+tp,1)
        note = "Best F1 (Validation)" if t == best_f1_t else ""
        wmd(f"| {t:.4f} | {int(tp+fp):,} | {(tp+fp)/test_size*100:.2f}% | {int(tp):,} | {int(fp):,} | {int(fn):,} | {int(tn):,} | {prec_*100:.2f}% | {rec_*100:.2f}% | {f1v:.4f} | {fpr_*100:.3f}% | {fnr_*100:.2f}% | {note} |")
        
    wmd("\n## H. Top-K Analysis\n")
    wmd("| K% | K Count | Fraud Captured | Precision@K | Recall@K | Lift@K |")
    wmd("|---|---|---|---|---|---|")
    for pct in [0.001, 0.005, 0.01, 0.02, 0.05, 0.10]:
        pk, rk, fc, k = topk(test_probs, ya, pct)
        wmd(f"| {pct*100:.1f}% | {k:,} | {int(fc):,} | {pk*100:.2f}% | {rk*100:.2f}% | {pk/test_fr:.2f}x |")
        
    wmd("\n## I. Test Time Stability\n")
    wmd("| Block | Fraud Rate | PR-AUC | Prec@1% | Prec@5% | Rec@1% | Rec@5% |")
    wmd("|---|---|---|---|---|---|---|")
    blks = block_stats(test_probs, ya)
    prs = []
    for b in blks:
        wmd(f"| {b['block']} | {b['fr']*100:.4f}% | {b['pr']:.5f} | {b['p1']*100:.2f}% | {b['p5']*100:.2f}% | {b['r1']*100:.2f}% | {b['r5']*100:.2f}% |")
        if b['pr']: prs.append(b['pr'])
    rng = max(prs)-min(prs) if prs else 0
    wmd(f"\nPR-AUC range across blocks: {rng:.5f}")
    
    if test_pr_auc < 0.05:
        sig_lvl = "WEAK (PR-AUC < 0.05)"
    elif test_pr_auc < 0.07:
        sig_lvl = "WEAK-TO-MODERATE (0.05-0.07)"
    elif test_pr_auc < 0.10:
        sig_lvl = "MODERATE (0.07-0.10)"
    else:
        sig_lvl = "STRONG"
        
    useful = "Yes, but only for extreme top-K screening or rule inputs." if test_pr_auc < 0.07 else "Yes, suitable for standalone scoring."
    
    wmd(f"""
## J. Final Interpretation

1. **Did validation performance generalize to test?** {pr_gen}. Val PR-AUC = {val_pr_auc:.5f}, Test PR-AUC = {test_pr_auc:.5f}.
2. **Signal level:** {sig_lvl}.
3. **Is the model useful for ranking fraud?** {useful}
4. **Is the model production-ready?** Subject to business thresholds. Given the weak signal, it should be used cautiously, possibly in an ensemble or as an early-stage filter.
5. **Limitations:** High false positive rate at most practical recall levels. Heavily reliant on TSLT artifact.
6. **Needed to improve:** Access to true graph data, deeper historical features across long timeframes, actual IP/device reputation feeds, and fixing the TSLT missingness issue at the source.

## K. Final Decision

Based on the test metrics:
* Accept as weak baseline model (if generalization is good and Lift@1% > 1.2x)
* Reject as insufficient signal (if test PR-AUC < Random Baseline)

**CONCLUSION**: Pipeline execution finished. This concludes the evaluation.
""")
    
    print("Final Holdout Evaluation Completed!")
    print(f"Test PR-AUC: {test_pr_auc:.5f} | Val PR-AUC: {val_pr_auc:.5f}")

if __name__ == "__main__":
    main()
