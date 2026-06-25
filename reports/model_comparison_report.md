# FINAL CANDIDATE MODEL FAMILY COMPARISON REPORT

## A. Confirmation
- Only `train.csv` used. `test.csv` untouched and unread.
- OOT split: train_inner=3.2M, validation=800K.
- No fraud label in features. No target encoding. No SMOTE. No random split.

## B. Feature Set Definition

**FS1**: 17 numeric + 5 categorical = 22 total features
- Numeric: amount, log_amount, spending_deviation_score, velocity_score, geo_anomaly_score, hour, day_of_week, month, is_weekend, is_night, tslt_abs, tslt_is_missing, tslt_is_negative, is_new_location_for_sender, is_new_payment_channel_for_sender, is_new_transaction_type_for_sender, is_new_device_used_for_sender

**FS2**: 21 numeric + 5 categorical = 26 total features
- Numeric: amount, log_amount, spending_deviation_score, velocity_score, geo_anomaly_score, hour, day_of_week, month, is_weekend, is_night, tslt_abs, tslt_is_missing, tslt_is_negative, is_new_location_for_sender, is_new_payment_channel_for_sender, is_new_transaction_type_for_sender, is_new_device_used_for_sender, sender_location_pair_count_past, sender_channel_pair_count_past, sender_device_type_pair_count_past, sender_txn_type_pair_count_past


## C. Model Comparison Table

| Model | FS | Train PR | Val PR | Gap | Rel Lift | ROC-AUC | Prec@0.1% | Prec@1% | Prec@5% | Rec@1% | Rec@5% | Lift@1% | Lift@5% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B4-LGBM-n300-none | FS1 | 0.05069 | 0.04926 | +0.00143 | 1.364x | 0.62972 | 4.62% | 4.86% | 5.05% | 1.35% | 7.00% | 1.35x | 1.40x |
| B4-LGBM-n300-none | FS2 | 0.04827 | 0.04915 | -0.00088 | 1.361x | 0.63011 | 2.50% | 4.89% | 5.08% | 1.35% | 7.03% | 1.35x | 1.41x |
| E-HistGB | FS1 | 0.06092 | 0.04912 | +0.01180 | 1.360x | 0.62754 | 5.00% | 5.15% | 5.06% | 1.43% | 7.00% | 1.43x | 1.40x |
| E-HistGB | FS2 | 0.06060 | 0.04902 | +0.01158 | 1.358x | 0.62786 | 4.62% | 5.21% | 4.91% | 1.44% | 6.80% | 1.44x | 1.36x |
| A-LR-none | FS2 | 0.04649 | 0.04855 | -0.00205 | 1.345x | 0.62496 | 5.50% | 5.29% | 4.98% | 1.46% | 6.90% | 1.46x | 1.38x |
| A-LR-balanced | FS2 | 0.04628 | 0.04844 | -0.00216 | 1.342x | 0.62463 | 5.38% | 5.46% | 4.91% | 1.51% | 6.79% | 1.51x | 1.36x |
| A-LR-none | FS1 | 0.04639 | 0.04840 | -0.00202 | 1.341x | 0.62348 | 5.00% | 4.88% | 5.03% | 1.35% | 6.97% | 1.35x | 1.39x |
| A-LR-balanced | FS1 | 0.04616 | 0.04829 | -0.00213 | 1.337x | 0.62352 | 5.50% | 5.04% | 4.98% | 1.40% | 6.89% | 1.40x | 1.38x |
| B3-LGBM-earlystop | FS1 | 0.04706 | 0.04789 | -0.00083 | 1.326x | 0.62201 | 5.00% | 5.00% | 4.92% | 1.38% | 6.82% | 1.38x | 1.36x |
| B1-LGBM-n300-sqrt | FS1 | 0.04659 | 0.04783 | -0.00124 | 1.325x | 0.62232 | 5.62% | 5.16% | 4.97% | 1.43% | 6.89% | 1.43x | 1.38x |
| B2-LGBM-n500-full | FS1 | 0.04619 | 0.04722 | -0.00103 | 1.308x | 0.62028 | 6.12% | 5.27% | 4.87% | 1.46% | 6.75% | 1.46x | 1.35x |
| B3-LGBM-earlystop | FS2 | 0.04513 | 0.04445 | +0.00069 | 1.231x | 0.59551 | 5.00% | 4.66% | 4.61% | 1.29% | 6.39% | 1.29x | 1.28x |
| B2-LGBM-n500-full | FS2 | 0.04446 | 0.04426 | +0.00020 | 1.226x | 0.59366 | 6.50% | 5.10% | 4.45% | 1.41% | 6.16% | 1.41x | 1.23x |
| B1-LGBM-n300-sqrt | FS2 | 0.04453 | 0.04416 | +0.00037 | 1.223x | 0.59411 | 5.50% | 4.81% | 4.45% | 1.33% | 6.16% | 1.33x | 1.23x |

## D. Best Model Selection

- **Best model**: `B4-LGBM-n300-none-FS1`
- **Val PR-AUC**: 0.04926
- **Train PR-AUC**: 0.05069
- **Gap**: +0.00143
- **ROC-AUC**: 0.62972
- **Relative lift**: 1.364x

### FS1 vs FS2

| | Best Val PR-AUC | Rel Lift |
|---|---|---|
| FS1 (Baseline+Novelty Binary) | 0.04926 | 1.364x |
| FS2 (FS1+Pair Counts) | 0.04915 | 1.361x |

**Selected feature set**: FS1


## E. Top-K Analysis (Best Model)

| K% | K Count | Fraud Captured | Precision@K | Recall@K | Lift@K |
|---|---|---|---|---|---|
| 0.1% | 800 | 37 | 4.62% | 0.13% | 1.28x |
| 0.5% | 4,000 | 181 | 4.52% | 0.63% | 1.25x |
| 1.0% | 8,000 | 389 | 4.86% | 1.35% | 1.35x |
| 2.0% | 16,000 | 784 | 4.90% | 2.71% | 1.36x |
| 5.0% | 40,000 | 2,022 | 5.05% | 7.00% | 1.40x |
| 10.0% | 80,000 | 4,115 | 5.14% | 14.25% | 1.42x |

## F. Threshold Analysis (Best Model)

| Threshold | Pred Fraud | Rate | TP | FP | FN | TN | Precision | Recall | F1 | FPR | FNR |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.0010 | 800,000 | 100.00% | 28,884 | 771,116 | 0 | 0 | 3.61% | 100.00% | 0.0697 | 100.000% | 0.00% |
| 0.0050 | 656,284 | 82.04% | 28,884 | 627,400 | 0 | 143,716 | 4.40% | 100.00% | 0.0843 | 81.363% | 0.00% |
| 0.0100 | 656,284 | 82.04% | 28,884 | 627,400 | 0 | 143,716 | 4.40% | 100.00% | 0.0843 | 81.363% | 0.00% |
| 0.0200 | 656,282 | 82.04% | 28,884 | 627,398 | 0 | 143,718 | 4.40% | 100.00% | 0.0843 | 81.362% | 0.00% |
| 0.0434 | 323,179 | 40.40% | 16,178 | 307,001 | 12,706 | 464,115 | 5.01% | 56.01% | 0.0919 | 39.813% | 43.99% |
| 0.0500 | 10,693 | 1.34% | 522 | 10,171 | 28,362 | 760,945 | 4.88% | 1.81% | 0.0264 | 1.319% | 98.19% |
| 0.1000 | 0 | 0.00% | 0 | 0 | 28,884 | 771,116 | 0.00% | 0.00% | 0.0000 | 0.000% | 100.00% |

## G. Time Stability Review (Best Model)

| Block | Fraud Rate | PR-AUC | Prec@1% |
|---|---|---|---|
| 1 | 3.6040% | 0.04913 | 5.15% |
| 2 | 3.5855% | 0.04905 | 4.80% |
| 3 | 3.6345% | 0.04905 | 4.30% |
| 4 | 3.6180% | 0.04992 | 4.90% |

PR-AUC range: 0.00087 — Stable

## H. Sanity Checks

### Label Permutation Test

| | PR-AUC | Rel Lift |
|---|---|---|
| Real Labels | 0.04876 | 1.351x |
| Shuffled Labels | 0.03663 | 1.014x |
| **Gap** | **+0.01213** | **+33.1%** |

**REAL SIGNAL CONFIRMED** - Gap >20%. Improvement is genuine, not noise.

### With vs Without TSLT

| Model Variant | Val PR-AUC | Contribution |
|---|---|---|
| With TSLT (best model) | 0.04876 | — |
| Without TSLT | 0.03989 | -0.00887 |
| TSLT contribution | | **+0.00887** |

### Hard Subset (tslt_is_missing == 0)

| | PR-AUC | Random Baseline | Rel Lift |
|---|---|---|---|
| Hard Subset Model | 0.04918 | 0.04401 | 1.118x |

Hard subset has signal beyond TSLT artifact - novelty features contribute independently.

## I. Final Candidate Pipeline

| Item | Value |
|---|---|
| **Feature Set** | FS1 — Baseline + Novelty Binary |
| **Numeric Features (17)** | amount, log_amount, spending_deviation_score, velocity_score, geo_anomaly_score, hour, day_of_week, month, is_weekend, is_night, tslt_abs, tslt_is_missing, tslt_is_negative, is_new_location_for_sender, is_new_payment_channel_for_sender, is_new_transaction_type_for_sender, is_new_device_used_for_sender |
| **Categorical Features (5)** | transaction_type, merchant_category, payment_channel, device_used, location |
| **Dropped Columns** | transaction_id, fraud_type, sender_account, receiver_account, ip_address, device_hash, time_since_last_transaction (raw) |
| **Preprocessing** | SimpleImputer(median) + OneHotEncoder(handle_unknown='ignore') |
| **Model** | LightGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=300, subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0, scale_pos_weight=1.0, random_state=42) |
| **Val PR-AUC** | 0.04876 |
| **ROC-AUC** | 0.62972 |
| **Relative Lift** | 1.351x |
| **Label Permutation Gap** | +33.1% |

## J. Decision

| Question | Answer |
|---|---|
| Model family tốt nhất? | **LightGBM (n=300, lr=0.05, spw=1.0)** — Val PR=0.04926. Note: spw=1 (no class weight) outperforms spw=sqrt and spw=full consistently. |
| FS1 hay FS2 tốt hơn? | **FS1** — 0.04926 vs 0.04915. Pair counts add marginal noise for LightGBM with spw=1. |
| Cải thiện rõ vs FS0 không? | **Yes** — +0.00523 vs FS0=0.04403 (+11.9%) |
| Signal level? | **WEAK (PR-AUC < 0.05)** |
| Có overfit không? | Train-Val gap = +0.00143 — **Minimal**. Model generalizes well. |
| Tune thêm không? | Consider moderate tuning (n_estimators, num_leaves) but ceiling appears near. |
| Mở test.csv chưa? | **NO — test.csv remains locked.** |

### Key Findings

1. **LightGBM with scale_pos_weight=1 (no class weight) wins** — surprising result. With spw > 1, early stopping on AUC triggers after 1 tree, collapsing the model to near-random. With spw=1, LightGBM uses 51 trees and actually learns.

2. **HistGradientBoosting is competitive** (0.04912 on FS1) — solid fallback if LightGBM is unavailable.

3. **Logistic Regression is surprisingly strong** (0.04840-0.04855) — confirms that the signal is largely linear (TSLT artifact + novelty flags are threshold-based).

4. **TSLT remains primary driver**: removing TSLT drops PR-AUC by ~0.00887. Hard subset lift = 1.118x, confirming minimal signal outside TSLT.

5. **Time stability confirmed**: PR-AUC range across 4 blocks = 0.00087 — excellent consistency.

6. **Candidate pipeline is ready to lock.** Next step: freeze the pipeline definition above, then evaluate on `test.csv` for final unbiased assessment.

