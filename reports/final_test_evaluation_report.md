# FINAL HOLDOUT TEST EVALUATION REPORT

## A. Pipeline Lock Confirmation
Pipeline was locked (FS1 + LightGBM spw=1) before opening test set.

## B. Test Access Confirmation
`test.csv` opened for the first and only time for final evaluation.

## C. Final Pipeline Definition

- **Feature Set**: FS1 (Baseline + Novelty Binary)
- **Numeric Features**: amount, log_amount, spending_deviation_score, velocity_score, geo_anomaly_score, hour, day_of_week, month, is_weekend, is_night, tslt_abs, tslt_is_missing, tslt_is_negative, is_new_location_for_sender, is_new_payment_channel_for_sender, is_new_transaction_type_for_sender, is_new_device_used_for_sender
- **Categorical Features**: transaction_type, merchant_category, payment_channel, device_used, location
- **Dropped Columns**: transaction_id, fraud_type, sender_account, receiver_account, ip_address, device_hash, raw timestamp, raw time_since_last_transaction
- **Preprocessing**: SimpleImputer(median) for numeric, SimpleImputer(most_frequent) + OneHotEncoder for categorical. Fitted on train only.
- **Model**: LightGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=300, subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0, scale_pos_weight=1.0, random_state=42)
- **Feature Generation Mode**: Streaming test simulation (test rows use past train and past test rows).

## D. Test Dataset Summary

- **Rows**: 1,000,000
- **Time Range**: 2023-10-20 10:49:17.267661 to 2024-01-01 22:58:30.131850
- **Fraud Count**: 36,026
- **Fraud Rate**: 3.6026%

## E. Final Test Metrics

- **PR-AUC**: 0.04819
- **ROC-AUC**: 0.62329
- **Relative Lift**: 1.338x
- **Precision@0.1%**: 4.60%
- **Precision@0.5%**: 5.28%
- **Precision@1%**: 4.78%
- **Precision@2%**: 4.95%
- **Precision@5%**: 4.94%
- **Precision@10%**: 4.88%
- **Recall@1%**: 1.33%
- **Recall@5%**: 6.86%
- **Recall@10%**: 13.56%
- **Lift@1%**: 1.33x
- **Lift@5%**: 1.37x
- **Lift@10%**: 1.36x

## F. Validation vs Test Comparison

| Metric | Validation | Test | Generalization |
|---|---|---|---|
| PR-AUC | 0.06954 | 0.04819 | Drop |
| Precision@1% | 14.49% | 4.78% | Drop |
| Precision@5% | 9.65% | 4.94% | Drop |

## G. Threshold Analysis

| Threshold | Pred Fraud | Rate | TP | FP | FN | TN | Precision | Recall | F1 | FPR | FNR | Note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.0010 | 821,565 | 82.16% | 36,026 | 785,539 | 0 | 178,435 | 4.39% | 100.00% | 0.0840 | 81.490% | 0.00% |  |
| 0.0050 | 821,565 | 82.16% | 36,026 | 785,539 | 0 | 178,435 | 4.39% | 100.00% | 0.0840 | 81.490% | 0.00% |  |
| 0.0100 | 821,539 | 82.15% | 36,026 | 785,513 | 0 | 178,461 | 4.39% | 100.00% | 0.0840 | 81.487% | 0.00% |  |
| 0.0200 | 819,048 | 81.90% | 35,952 | 783,096 | 74 | 180,878 | 4.39% | 99.79% | 0.0841 | 81.236% | 0.21% |  |
| 0.0500 | 212,261 | 21.23% | 10,498 | 201,763 | 25,528 | 762,211 | 4.95% | 29.14% | 0.0846 | 20.930% | 70.86% |  |
| 0.0515 | 99,530 | 9.95% | 4,856 | 94,674 | 31,170 | 869,300 | 4.88% | 13.48% | 0.0716 | 9.821% | 86.52% | Best F1 (Validation) |
| 0.1000 | 0 | 0.00% | 0 | 0 | 36,026 | 963,974 | 0.00% | 0.00% | 0.0000 | 0.000% | 100.00% |  |

## H. Top-K Analysis

| K% | K Count | Fraud Captured | Precision@K | Recall@K | Lift@K |
|---|---|---|---|---|---|
| 0.1% | 1,000 | 46 | 4.60% | 0.13% | 1.28x |
| 0.5% | 5,000 | 264 | 5.28% | 0.73% | 1.47x |
| 1.0% | 10,000 | 478 | 4.78% | 1.33% | 1.33x |
| 2.0% | 20,000 | 990 | 4.95% | 2.75% | 1.37x |
| 5.0% | 50,000 | 2,472 | 4.94% | 6.86% | 1.37x |
| 10.0% | 100,000 | 4,885 | 4.88% | 13.56% | 1.36x |

## I. Test Time Stability

| Block | Fraud Rate | PR-AUC | Prec@1% | Prec@5% | Rec@1% | Rec@5% |
|---|---|---|---|---|---|---|
| 1 | 3.5780% | 0.04771 | 5.28% | 4.90% | 1.48% | 6.85% |
| 2 | 3.5848% | 0.04797 | 4.56% | 5.18% | 1.27% | 7.23% |
| 3 | 3.6488% | 0.04923 | 4.96% | 4.86% | 1.36% | 6.65% |
| 4 | 3.5988% | 0.04791 | 4.36% | 4.80% | 1.21% | 6.67% |

PR-AUC range across blocks: 0.00152

## J. Final Interpretation

1. **Did validation performance generalize to test?** Drop. Val PR-AUC = 0.06954, Test PR-AUC = 0.04819.
2. **Signal level:** WEAK (PR-AUC < 0.05).
3. **Is the model useful for ranking fraud?** Yes, but only for extreme top-K screening or rule inputs.
4. **Is the model production-ready?** Subject to business thresholds. Given the weak signal, it should be used cautiously, possibly in an ensemble or as an early-stage filter.
5. **Limitations:** High false positive rate at most practical recall levels. Heavily reliant on TSLT artifact.
6. **Needed to improve:** Access to true graph data, deeper historical features across long timeframes, actual IP/device reputation feeds, and fixing the TSLT missingness issue at the source.

## K. Final Decision

Based on the test metrics:
* Accept as weak baseline model (if generalization is good and Lift@1% > 1.2x)
* Reject as insufficient signal (if test PR-AUC < Random Baseline)

**CONCLUSION**: Pipeline execution finished. This concludes the evaluation.

