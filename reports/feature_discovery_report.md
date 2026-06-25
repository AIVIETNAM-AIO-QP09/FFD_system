# FEATURE DISCOVERY & STRONG FEATURE MINING REPORT

## A. Objective
Discover whether genuine predictive signal exists in this dataset beyond the tslt_is_missing artifact, using novelty, all-past frequency, sender-relative amount, lifecycle, and unusual-hour features.
Only `train.csv` used. `test.csv` untouched.

## E. Selected P0/P1 Features
Building the following feature families:
- Novelty: is_new_receiver_for_sender, is_new_device_for_sender, is_new_merchant_category_for_sender, is_new_payment_channel_for_sender, is_new_location_for_sender
- All-past frequency: sender_tx_index, receiver_tx_index, sender_seen_count_all_past, receiver_seen_count_all_past, receiver_in_degree_all_past, sender_out_degree_all_past
- Sender-relative amount: sender_amount_zscore_past, is_sender_new_max_amount, amount_ratio_to_sender_mean
- Lifecycle: sender_days_since_first_seen, sender_days_since_last_seen, receiver_days_since_first_seen
- Unusual-hour: hour_deviation_from_sender_mean, is_unusual_hour_for_sender

IP/device rolling features REJECTED due to near-unique cardinality (>3M unique values on 4M rows).

## F. Leakage Safety Design
- All features use `cumcount()` or expanding transform with `shift(1)`, strictly counting only rows BEFORE the current row in chronological order within each entity group.
- No `is_fraud` label used in feature computation.
- Preprocessing (Imputer) fit strictly on train_inner (first 3.2M rows), transform on validation.

## G. New Feature Diagnostics

| Feature | Null Rate | Zero Rate | PR-AUC | Rel Lift | Signal |
|---|---|---|---|---|---|
| sender_tx_index | 0.00% | 1.90% | 0.03842 | 1.064x | ⚠️ |
| is_sender_first_tx | 0.00% | 98.10% | 0.03635 | 1.007x | ❌ |
| sender_seen_count_all_past | 0.00% | 1.90% | 0.03842 | 1.064x | ⚠️ |
| receiver_tx_index | 0.00% | 1.91% | 0.03618 | 1.002x | ❌ |
| is_receiver_first_tx | 0.00% | 98.09% | 0.03614 | 1.001x | ❌ |
| receiver_seen_count_all_past | 0.00% | 1.91% | 0.03618 | 1.002x | ❌ |
| sender_receiver_pair_count_past | 0.00% | 100.00% | 0.03611 | 1.000x | ❌ |
| is_new_receiver_for_sender | 0.00% | 0.00% | 0.03611 | 1.000x | ❌ |
| sender_device_pair_count_past | 0.00% | 100.00% | 0.03611 | 1.000x | ❌ |
| is_new_device_for_sender | 0.00% | 0.00% | 0.03611 | 1.000x | ❌ |
| is_new_merchant_category_for_sender | 0.00% | 39.22% | 0.03655 | 1.012x | ❌ |
| is_new_payment_channel_for_sender | 0.00% | 63.05% | 0.03687 | 1.021x | ⚠️ |
| is_new_location_for_sender | 0.00% | 39.38% | 0.03812 | 1.056x | ⚠️ |
| receiver_in_degree_all_past | 0.00% | 1.91% | 0.03618 | 1.002x | ❌ |
| is_new_sender_for_receiver | 0.00% | 0.00% | 0.03611 | 1.000x | ❌ |
| sender_out_degree_all_past | 0.00% | 1.90% | 0.03842 | 1.064x | ⚠️ |
| sender_amount_mean_past | 1.90% | 1.90% | 0.03611 | 1.000x | ❌ |
| sender_amount_std_past | 1.90% | 9.34% | 0.03728 | 1.032x | ⚠️ |
| sender_amount_max_past | 1.90% | 1.90% | 0.03736 | 1.035x | ⚠️ |
| sender_amount_zscore_past | 1.90% | 1.90% | 0.03566 | 0.988x | ❌ |
| is_sender_new_max_amount | 0.00% | 75.35% | 0.03664 | 1.015x | ❌ |
| amount_ratio_to_sender_mean | 1.90% | 1.90% | 0.03671 | 1.017x | ❌ |
| sender_days_since_first_seen | 0.00% | 1.90% | 0.03698 | 1.024x | ⚠️ |
| sender_days_since_last_seen | 1.90% | 1.90% | 0.03730 | 1.033x | ⚠️ |
| receiver_days_since_first_seen | 0.00% | 1.91% | 0.03610 | 1.000x | ❌ |
| sender_mean_hour_past | 1.90% | 2.22% | 0.03552 | 0.984x | ❌ |
| hour_deviation_from_sender_mean | 1.90% | 3.25% | 0.03654 | 1.012x | ❌ |
| is_unusual_hour_for_sender | 0.00% | 51.09% | 0.03613 | 1.001x | ❌ |

## H. Modeling Experiment Results

| Experiment | Train PR-AUC | Val PR-AUC | Gap | Diff vs Baseline | Rel Lift | Prec@1% | Prec@5% |
|---|---|---|---|---|---|---|---|
| Exp 0 — Baseline | 0.05929 | 0.04403 | +0.01526 | +0.00000 | 1.219x | 4.61% | 4.35% |
| Exp 1 — Baseline + Novelty | 0.05606 | 0.04907 | +0.00698 | +0.00505 | 1.359x | 4.88% | 5.06% |
| Exp 2 — Baseline + Sender Amount | 0.05981 | 0.04394 | +0.01587 | -0.00009 | 1.217x | 4.10% | 4.42% |
| Exp 3 — Baseline + Lifecycle | 0.05990 | 0.04359 | +0.01631 | -0.00044 | 1.207x | 3.92% | 4.22% |
| Exp 4 — Baseline + Unusual Hour | 0.06014 | 0.04374 | +0.01640 | -0.00029 | 1.211x | 4.50% | 4.29% |
| Exp 5 — All New Features | 0.05943 | 0.04852 | +0.01091 | +0.00449 | 1.344x | 4.99% | 5.00% |
| Exp 6 — Hard Subset + All Features | 0.06120 | 0.04845 | +0.01275 | +0.00443 | 1.342x | 5.23% | 4.98% |

## J. Time Stability Review

| Block | Fraud Rate | PR-AUC | Prec@1% |
|---|---|---|---|
| 1 | 3.6040% | 0.04839 | 4.50% |
| 2 | 3.5855% | 0.04961 | 4.80% |
| 3 | 3.6345% | 0.04902 | 5.00% |
| 4 | 3.6180% | 0.04937 | 4.75% |

## Top 20 Features (Best Experiment)

| Rank | Feature | Importance |
|---|---|---|
| 1 | tslt_abs | 428 |
| 2 | spending_deviation_score | 369 |
| 3 | amount | 351 |
| 4 | geo_anomaly_score | 314 |
| 5 | hour | 206 |
| 6 | velocity_score | 186 |
| 7 | sender_tx_index | 126 |
| 8 | receiver_tx_index | 104 |
| 9 | month | 93 |
| 10 | day_of_week | 92 |
| 11 | tslt_is_missing | 76 |
| 12 | tslt_is_negative | 51 |
| 13 | is_new_location_for_sender | 42 |
| 14 | payment_channel_UPI | 29 |
| 15 | device_used_mobile | 28 |
| 16 | location_Tokyo | 26 |
| 17 | location_Toronto | 26 |
| 18 | merchant_category_travel | 24 |
| 19 | merchant_category_entertainment | 23 |
| 20 | location_Singapore | 22 |

## L. Final Recommendation

### Key Metrics
- **Random Baseline PR-AUC**: 0.03610
- **Previous Baseline PR-AUC**: ~0.04454
- **Best New Experiment PR-AUC**: 0.04907
- **Best Experiment**: Exp 1 — Baseline + Novelty
- **Improvement over previous baseline**: +0.00453

### Conclusions

1. **Tìm được feature mạnh mới không?** Yes — improvement recorded.
2. **PR-AUC vượt rõ baseline không?** Yes
3. **Có overfit không?** Minimal
4. **Có nên tune model chưa?** Consider tuning if signal is confirmed.
5. **Có được mở test.csv chưa?** **NO — test.csv remains locked.**

### Dataset Signal Verdict
🟡 Some improvement detected. Investigate further before tuning.

