# CANDIDATE FEATURE STABILIZATION & CONTROLLED TUNING REPORT

## A. Objective
Stabilize best feature set from novelty/pair-familiarity round, prune noise, confirm improvement is real, then perform controlled LightGBM tuning.

## B. Confirmation
- Only `train.csv` used. `test.csv` untouched.
- OOT split: train_inner = 3.2M rows, validation = 800K rows.
- No fraud label in feature computation. No target encoding. No SMOTE.

## C. Feature Set Comparison (FS0–FS5)

| Feature Set | #Feats | Train PR | Val PR | Gap | Diff vs FS0 | Rel Lift | Prec@0.1% | Prec@1% | Prec@5% | Rec@1% | Rec@5% |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FS0 — Baseline | 13 | 0.05929 | 0.04403 | +0.01526 | +0.00000 | 1.219x | 6.75% | 4.61% | 4.35% | 1.28% | 6.03% |
| FS1 — Baseline + Novelty Binary | 17 | 0.05546 | 0.04897 | +0.00649 | +0.00494 | 1.356x | 4.62% | 4.72% | 4.89% | 1.31% | 6.77% |
| FS2 — Baseline + Pair Counts | 17 | 0.05585 | 0.04867 | +0.00719 | +0.00464 | 1.348x | 3.75% | 4.59% | 4.90% | 1.27% | 6.78% |
| FS3 — Baseline + Sender Maturity | 21 | 0.05819 | 0.04416 | +0.01403 | +0.00013 | 1.223x | 4.88% | 4.58% | 4.48% | 1.27% | 6.20% |
| FS4 — Baseline + Best Selected | 29 | 0.05702 | 0.04866 | +0.00837 | +0.00463 | 1.348x | 5.12% | 4.72% | 4.98% | 1.31% | 6.89% |
| FS5 — FS4 minus TSLT | 26 | 0.04684 | 0.04067 | +0.00617 | -0.00336 | 1.127x | 4.12% | 4.09% | 4.49% | 1.13% | 6.22% |

## E. Group Ablation Results

| Group Removed | Val PR | Diff vs Full | Impact |
|---|---|---|---|
| Full FS4 | 0.04866 | +0.00000 | 🟢 LOW |
| Remove novelty binary | 0.04866 | +0.00000 | 🟢 LOW |
| Remove pair counts | 0.04775 | -0.00091 | 🟡 MEDIUM |
| Remove sender maturity | 0.04867 | +0.00001 | 🟢 LOW |
| Remove TSLT features | 0.04067 | -0.00798 | 🔴 HIGH |
| Remove behavioral scores | 0.04865 | -0.00000 | 🟢 LOW |

## F. Controlled Tuning Results (Top 10 Configs)

| Rank | leaves | depth | lr | mcs | reg_alpha | reg_lambda | spw | n_trees | Train PR | Val PR | Gap | Prec@1% | Prec@5% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 63 | -1 | 0.05 | 300 | 0.1 | 1 | 5.2 | 1 | 0.04706 | 0.04789 | -0.00083 | 5.00% | 4.92% |
| 2 | 63 | -1 | 0.1 | 300 | 0.1 | 1 | 5.2 | 1 | 0.04706 | 0.04789 | -0.00083 | 5.00% | 4.92% |
| 3 | 63 | -1 | 0.05 | 1000 | 0.1 | 1 | 5.2 | 1 | 0.04713 | 0.04787 | -0.00075 | 4.96% | 4.99% |
| 4 | 63 | -1 | 0.1 | 1000 | 0.1 | 1 | 5.2 | 1 | 0.04713 | 0.04787 | -0.00075 | 4.96% | 4.99% |
| 5 | 31 | -1 | 0.05 | 300 | 0.1 | 1 | 5.2 | 1 | 0.04659 | 0.04783 | -0.00124 | 5.16% | 4.97% |
| 6 | 31 | -1 | 0.05 | 1000 | 0.1 | 1 | 5.2 | 1 | 0.04656 | 0.04783 | -0.00127 | 5.16% | 4.97% |
| 7 | 31 | -1 | 0.1 | 300 | 0.1 | 1 | 5.2 | 1 | 0.04659 | 0.04783 | -0.00124 | 5.16% | 4.97% |
| 8 | 31 | -1 | 0.1 | 1000 | 0.1 | 1 | 5.2 | 1 | 0.04656 | 0.04783 | -0.00127 | 5.16% | 4.97% |
| 9 | 31 | 5 | 0.05 | 300 | 0.1 | 1 | 5.2 | 1 | 0.04632 | 0.04782 | -0.00150 | 4.90% | 5.16% |
| 10 | 31 | 5 | 0.1 | 300 | 0.1 | 1 | 5.2 | 1 | 0.04632 | 0.04782 | -0.00150 | 4.90% | 5.16% |

## G. Best Model Metrics

- **Val PR-AUC**: 0.04789
- **Train PR-AUC**: 0.04706
- **Gap**: -0.00083
- **ROC-AUC**: 0.62201
- **Precision@0.1%**: 5.00%
- **Precision@1%**: 5.00%
- **Precision@5%**: 4.92%
- **Recall@1%**: 1.38%
- **Recall@5%**: 6.82%
- **Best n_estimators** (early stopping): 1

## H. Threshold Analysis (Best Tuned Model)

| Threshold | Pred Fraud | Pred Rate | TP | FP | FN | TN | Precision | Recall | F1 | FPR | FNR |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.0010 | 800,000 | 100.00% | 28,884 | 771,116 | 0 | 0 | 3.61% | 100.00% | 0.0697 | 100.000% | 0.00% |
| 0.0050 | 800,000 | 100.00% | 28,884 | 771,116 | 0 | 0 | 3.61% | 100.00% | 0.0697 | 100.000% | 0.00% |
| 0.0100 | 800,000 | 100.00% | 28,884 | 771,116 | 0 | 0 | 3.61% | 100.00% | 0.0697 | 100.000% | 0.00% |
| 0.0200 | 800,000 | 100.00% | 28,884 | 771,116 | 0 | 0 | 3.61% | 100.00% | 0.0697 | 100.000% | 0.00% |
| 0.0446 | 377,802 | 47.23% | 18,420 | 359,382 | 10,464 | 411,734 | 4.88% | 63.77% | 0.0906 | 46.605% | 36.23% |
| 0.0500 | 0 | 0.00% | 0 | 0 | 28,884 | 771,116 | 0.00% | 0.00% | 0.0000 | 0.000% | 100.00% |
| 0.1000 | 0 | 0.00% | 0 | 0 | 28,884 | 771,116 | 0.00% | 0.00% | 0.0000 | 0.000% | 100.00% |
| 0.2000 | 0 | 0.00% | 0 | 0 | 28,884 | 771,116 | 0.00% | 0.00% | 0.0000 | 0.000% | 100.00% |
| 0.5000 | 0 | 0.00% | 0 | 0 | 28,884 | 771,116 | 0.00% | 0.00% | 0.0000 | 0.000% | 100.00% |

## I. Top-K Analysis

| K% | K Count | Fraud Captured | Precision@K | Recall@K | Lift@K |
|---|---|---|---|---|---|
| 0.1% | 800 | 40 | 5.00% | 0.14% | 1.38x |
| 0.5% | 4,000 | 196 | 4.90% | 0.68% | 1.36x |
| 1.0% | 8,000 | 400 | 5.00% | 1.38% | 1.38x |
| 2.0% | 16,000 | 812 | 5.08% | 2.81% | 1.41x |
| 5.0% | 40,000 | 1,969 | 4.92% | 6.82% | 1.36x |
| 10.0% | 80,000 | 4,040 | 5.05% | 13.99% | 1.40x |

## J. Time Stability Review

| Block | Fraud Rate | PR-AUC | Prec@1% | Prec@5% | Rec@1% |
|---|---|---|---|---|---|
| 1 | 3.6040% | 0.04741 | 5.55% | 4.79% | 1.54% |
| 2 | 3.5855% | 0.04786 | 4.70% | 4.85% | 1.31% |
| 3 | 3.6345% | 0.04777 | 5.20% | 5.18% | 1.43% |
| 4 | 3.6180% | 0.04864 | 4.40% | 5.23% | 1.22% |

PR-AUC range across blocks: 0.00122 (Stable ✅)

## K. Sanity Checks

### Label Permutation Test

| | PR-AUC | Rel Lift | Gap vs Shuffle |
|---|---|---|---|
| Real Labels | 0.04789 | 1.326x | — |
| Shuffled Labels | 0.03618 | 1.002x | — |
| **Difference** | **+0.01171** | | **+32.4%** |

🟢 **REAL SIGNAL**: Real labels significantly outperform shuffled. Improvement is genuine.

### With / Without TSLT

| Model | Val PR-AUC | Rel Lift |
|---|---|---|
| With TSLT (best tuned) | 0.04789 | 1.326x |
| Without TSLT | 0.03943 | 1.092x |
| TSLT contribution | +0.00846 | |

### Hard Subset (tslt_is_missing == 0)

| | PR-AUC | Random Baseline | Rel Lift |
|---|---|---|---|
| Hard Subset Model | 0.04781 | 0.04401 | 1.086x |

## L. Final Recommendation

### Summary
- **Random Baseline PR-AUC**: 0.03610
- **Previous Baseline PR-AUC (FS0 baseline)**: 0.04403
- **Best Feature Set**: FS1 — Baseline + Novelty Binary → Val PR=0.04897
- **Best Tuned Model Val PR-AUC**: 0.04789
- **Improvement vs FS0**: +0.00386
- **Improvement vs random**: 1.326x

### Answers
1. **Feature set tốt nhất**: FS1 — Baseline + Novelty Binary
2. **Model tốt nhất**: LightGBM num_leaves=63, depth=-1, lr=0.05, mcs=300, reg_lambda=1, spw=5.2, n_trees=1
3. **Improvement có thật không?** Yes — label permutation confirms gap of 32.4%
4. **Signal level**: **WEAK** (PR-AUC < 0.05)
5. **Có overfit không?** Minimal — train/val gap is reasonable.
6. **Có nên tiếp tục feature engineering không?** Yes — more signal discovery needed.
7. **Có nên tune thêm không?** Not yet — signal too weak.
8. **Có được mở test.csv chưa?** **NO — test.csv remains locked. Pipeline must be frozen first.**

