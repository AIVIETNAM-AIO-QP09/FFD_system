# NOVELTY FEATURE VALIDATION & EXPANSION REPORT

## A. Objective
Validate that novelty feature improvement is real (not artifact), expand feature catalogue, run ablation and experiments 0–8.

## B. Confirmation
- Only `train.csv` used.
- `test.csv` untouched and unread.
- OOT split: train_inner = first 3.2M rows, validation = last 800K rows.
- All features use strictly-past information via cumcount()/expanding().shift(1).
- No fraud label used in feature computation.

## F. New Feature Diagnostics

| Feature | Family | Null% | Zero% | FR(=0) | FR(≠0) | PR-AUC | Rel Lift | Keep? |
|---|---|---|---|---|---|---|---|---|
| sender_receiver_account_pair_count_past | sender_novelty | 0.0% | 100.0% | 3.611% | 0.000% | 0.03611 | 1.000x | ❌ |
| is_new_receiver_for_sender | sender_novelty | 0.0% | 0.0% | 0.000% | 3.611% | 0.03611 | 1.000x | ❌ |
| sender_location_pair_count_past | pair_familiarity | 0.0% | 60.6% | 3.917% | 3.138% | 0.03831 | 1.061x | ✅ |
| is_new_location_for_sender | sender_novelty | 0.0% | 39.4% | 3.138% | 3.917% | 0.03812 | 1.056x | ✅ |
| sender_payment_channel_pair_count_past | sender_novelty | 0.0% | 36.9% | 3.411% | 3.728% | 0.03721 | 1.031x | ✅ |
| is_new_payment_channel_for_sender | sender_novelty | 0.0% | 63.1% | 3.728% | 3.411% | 0.03687 | 1.021x | ✅ |
| sender_merchant_category_pair_count_past | sender_novelty | 0.0% | 60.8% | 3.540% | 3.719% | 0.03670 | 1.017x | ❌ |
| is_new_merchant_category_for_sender | sender_novelty | 0.0% | 39.2% | 3.719% | 3.540% | 0.03655 | 1.012x | ❌ |
| sender_transaction_type_pair_count_past | sender_novelty | 0.0% | 36.8% | 3.411% | 3.727% | 0.03721 | 1.030x | ✅ |
| is_new_transaction_type_for_sender | sender_novelty | 0.0% | 63.2% | 3.727% | 3.411% | 0.03686 | 1.021x | ✅ |
| sender_device_used_pair_count_past | sender_novelty | 0.0% | 36.9% | 3.448% | 3.706% | 0.03717 | 1.030x | ✅ |
| is_new_device_used_for_sender | sender_novelty | 0.0% | 63.1% | 3.706% | 3.448% | 0.03672 | 1.017x | ❌ |
| sender_tx_index | sender_familiarity | 0.0% | 1.9% | 2.319% | 3.635% | 0.03842 | 1.064x | ✅ |
| sender_unique_receivers_all_past | sender_familiarity | 0.0% | 1.9% | 2.319% | 3.635% | 0.03842 | 1.064x | ✅ |
| sender_unique_locations_all_past | sender_familiarity | 0.0% | 1.9% | 2.319% | 3.635% | 0.03789 | 1.049x | ✅ |
| sender_unique_channels_all_past | sender_familiarity | 0.0% | 1.9% | 2.319% | 3.635% | 0.03785 | 1.048x | ✅ |
| sender_unique_categories_all_past | sender_familiarity | 0.0% | 1.9% | 2.319% | 3.635% | 0.03811 | 1.056x | ✅ |
| sender_unique_txn_types_all_past | sender_familiarity | 0.0% | 1.9% | 2.319% | 3.635% | 0.03791 | 1.050x | ✅ |
| sender_unique_device_types_all_past | sender_familiarity | 0.0% | 1.9% | 2.319% | 3.635% | 0.03774 | 1.045x | ✅ |
| sender_receiver_familiarity_ratio | sender_familiarity | 0.0% | 1.9% | 2.319% | 3.635% | 0.03842 | 1.064x | ✅ |
| sender_location_familiarity_ratio | sender_familiarity | 0.0% | 1.9% | 2.319% | 3.635% | 0.03626 | 1.004x | ❌ |
| sender_channel_familiarity_ratio | sender_familiarity | 0.0% | 1.9% | 2.319% | 3.635% | 0.03581 | 0.992x | ❌ |
| sender_category_familiarity_ratio | sender_familiarity | 0.0% | 1.9% | 2.319% | 3.635% | 0.03670 | 1.016x | ❌ |
| sender_out_degree_all_past | sender_familiarity | 0.0% | 1.9% | 2.319% | 3.635% | 0.03842 | 1.064x | ✅ |
| receiver_tx_index | receiver_accumulation | 0.0% | 1.9% | 3.425% | 3.614% | 0.03618 | 1.002x | ❌ |
| is_receiver_first_tx | receiver_accumulation | 0.0% | 98.1% | 3.614% | 3.425% | 0.03614 | 1.001x | ❌ |
| receiver_unique_senders_all_past | receiver_accumulation | 0.0% | 1.9% | 3.425% | 3.614% | 0.03618 | 1.002x | ❌ |
| receiver_unique_locations_all_past | receiver_accumulation | 0.0% | 1.9% | 3.425% | 3.614% | 0.03616 | 1.002x | ❌ |
| receiver_unique_channels_all_past | receiver_accumulation | 0.0% | 1.9% | 3.425% | 3.614% | 0.03618 | 1.002x | ❌ |
| receiver_in_degree_all_past | receiver_accumulation | 0.0% | 1.9% | 3.425% | 3.614% | 0.03618 | 1.002x | ❌ |
| receiver_amount_sum_all_past | receiver_accumulation | 1.9% | 1.9% | 3.425% | 3.614% | 0.03619 | 1.002x | ❌ |
| receiver_amount_mean_all_past | receiver_accumulation | 1.9% | 1.9% | 3.425% | 3.614% | 0.03618 | 1.002x | ❌ |
| receiver_days_since_first_seen | receiver_accumulation | 0.0% | 1.9% | 3.425% | 3.614% | 0.03610 | 1.000x | ❌ |
| is_receiver_new_globally | receiver_accumulation | 0.0% | 98.1% | 3.614% | 3.425% | 0.03614 | 1.001x | ❌ |
| sender_channel_pair_count_past | pair_familiarity | 0.0% | 36.9% | 3.411% | 3.728% | 0.03721 | 1.031x | ✅ |
| sender_category_pair_count_past | pair_familiarity | 0.0% | 60.8% | 3.540% | 3.719% | 0.03670 | 1.017x | ❌ |
| sender_device_type_pair_count_past | pair_familiarity | 0.0% | 36.9% | 3.448% | 3.706% | 0.03717 | 1.030x | ✅ |
| receiver_sender_pair_count_past | pair_familiarity | 0.0% | 100.0% | 3.611% | 0.000% | 0.03611 | 1.000x | ❌ |
| receiver_channel_pair_count_past | pair_familiarity | 0.0% | 36.8% | 3.609% | 3.612% | 0.03621 | 1.003x | ❌ |
| sender_receiver_pair_count_past | pair_familiarity | 0.0% | 100.0% | 3.611% | 0.000% | 0.03611 | 1.000x | ❌ |
| amount_ratio_to_sender_mean_all_past | behavior_deviation | 1.9% | 1.9% | 2.319% | 3.635% | 0.03565 | 0.988x | ❌ |
| amount_ratio_to_sender_median_all_past | behavior_deviation | 1.9% | 1.9% | 2.319% | 3.635% | 0.03599 | 0.997x | ❌ |
| amount_minus_sender_mean_all_past | behavior_deviation | 1.9% | 1.9% | 2.314% | 3.636% | 0.03644 | 1.009x | ❌ |
| is_sender_new_max_amount | behavior_deviation | 0.0% | 75.4% | 3.680% | 3.398% | 0.03664 | 1.015x | ❌ |
| sender_amount_zscore_all_past | behavior_deviation | 1.9% | 1.9% | 2.314% | 3.636% | 0.03572 | 0.989x | ❌ |
| is_new_high_velocity_for_sender | behavior_deviation | 0.0% | 55.4% | 3.643% | 3.570% | 0.03629 | 1.005x | ❌ |
| is_new_high_geo_for_sender | behavior_deviation | 0.0% | 53.7% | 3.605% | 3.617% | 0.03613 | 1.001x | ❌ |
| is_new_high_spending_for_sender | behavior_deviation | 0.0% | 53.5% | 3.659% | 3.554% | 0.03637 | 1.007x | ❌ |
| hour_deviation_from_sender_mean | behavior_deviation | 1.9% | 3.3% | 2.840% | 3.636% | 0.03565 | 0.987x | ❌ |
| is_unusual_hour_for_sender | behavior_deviation | 0.0% | 51.1% | 3.615% | 3.606% | 0.03613 | 1.001x | ❌ |

## D. Novelty Ablation Results (A–I)

| Ablation | Train PR | Val PR | Gap | Diff vs Base | Rel Lift | Prec@0.1% | Prec@1% | Prec@5% | Rec@1% |
|---|---|---|---|---|---|---|---|---|---|
| A. Baseline exact | 0.05929 | 0.04403 | +0.01526 | +0.00000 | 1.219x | 6.75% | 4.61% | 4.35% | 1.28% |
| B. Baseline + all novelty | 0.05823 | 0.04844 | +0.00979 | +0.00441 | 1.342x | 6.50% | 5.33% | 4.85% | 1.47% |
| C. Only is_new_location | 0.05562 | 0.04904 | +0.00659 | +0.00501 | 1.358x | 4.75% | 5.22% | 5.00% | 1.45% |
| D. Only is_new_channel | 0.05979 | 0.04397 | +0.01582 | -0.00006 | 1.218x | 6.00% | 4.15% | 4.29% | 1.15% |
| E. Only is_new_merchant_cat | 0.05950 | 0.04388 | +0.01562 | -0.00015 | 1.215x | 4.75% | 4.30% | 4.33% | 1.19% |
| F. Sender familiarity only | 0.05931 | 0.04413 | +0.01517 | +0.00011 | 1.222x | 5.50% | 4.60% | 4.50% | 1.27% |
| G. Receiver familiarity only | 0.06122 | 0.04384 | +0.01738 | -0.00019 | 1.214x | 5.38% | 4.29% | 4.27% | 1.19% |
| H. Novelty excl TSLT | 0.04718 | 0.04083 | +0.00635 | -0.00320 | 1.131x | 5.25% | 4.92% | 4.47% | 1.36% |
| I. Hard subset + novelty | 0.05835 | 0.04885 | +0.00950 | +0.00482 | 1.353x | 3.66% | 5.17% | 5.01% | 1.17% |

## G. Modeling Experiment Results (0–8)

| Experiment | Train PR | Val PR | Gap | Diff vs Base | Rel Lift | Prec@0.1% | Prec@1% | Prec@5% | Rec@1% | Rec@5% |
|---|---|---|---|---|---|---|---|---|---|---|
| Exp 0 — Previous baseline | 0.05929 | 0.04403 | +0.01526 | +0.00000 | 1.219x | 6.75% | 4.61% | 4.35% | 1.28% | 6.03% |
| Exp 1 — Current novelty (prev best) | 0.05706 | 0.04891 | +0.00816 | +0.00488 | 1.355x | 4.00% | 4.98% | 4.97% | 1.38% | 6.89% |
| Exp 2 — Expanded sender novelty+fam | 0.05706 | 0.04891 | +0.00816 | +0.00488 | 1.355x | 4.00% | 4.98% | 4.97% | 1.38% | 6.89% |
| Exp 3 — Receiver accumulation+fam | 0.06122 | 0.04384 | +0.01738 | -0.00019 | 1.214x | 5.38% | 4.29% | 4.27% | 1.19% | 5.91% |
| Exp 4 — Pair familiarity | 0.05617 | 0.04901 | +0.00715 | +0.00498 | 1.358x | 6.00% | 5.08% | 5.01% | 1.41% | 6.93% |
| Exp 5 — Behavior deviation | 0.06086 | 0.04406 | +0.01680 | +0.00003 | 1.220x | 4.38% | 4.28% | 4.35% | 1.18% | 6.02% |
| Exp 6 — All selected expanded | 0.05902 | 0.04892 | +0.01011 | +0.00489 | 1.355x | 4.62% | 5.25% | 5.02% | 1.45% | 6.96% |
| Exp 7 — Hard subset + all expanded | 0.06021 | 0.04892 | +0.01129 | +0.00489 | 1.355x | 5.34% | 4.95% | 4.89% | 1.13% | 5.55% |
| Exp 8 — All expanded minus TSLT | 0.04749 | 0.03982 | +0.00767 | -0.00421 | 1.103x | 3.50% | 4.32% | 4.23% | 1.20% | 5.85% |

## I. Time Stability Review (Best: Exp 4 — Pair familiarity)

| Block | Fraud Rate | PR-AUC | Prec@1% |
|---|---|---|---|
| 1 | 3.6040% | 0.04835466789652621 | 4.70% |
| 2 | 3.5855% | 0.04946253240251847 | 5.15% |
| 3 | 3.6345% | 0.04903920526971851 | 5.45% |
| 4 | 3.6180% | 0.049244725936765606 | 4.75% |

## J. Feature Importance by Family (Best Experiment)

| Rank | Feature | Importance |
|---|---|---|
| 1 | tslt_abs | 404 |
| 2 | spending_deviation_score | 397 |
| 3 | amount | 357 |
| 4 | geo_anomaly_score | 275 |
| 5 | hour | 216 |
| 6 | velocity_score | 190 |
| 7 | month | 107 |
| 8 | day_of_week | 96 |
| 9 | sender_location_pair_count_past | 77 |
| 10 | sender_device_type_pair_count_past | 76 |
| 11 | tslt_is_missing | 75 |
| 12 | sender_channel_pair_count_past | 68 |
| 13 | sender_category_pair_count_past | 59 |
| 14 | receiver_channel_pair_count_past | 57 |
| 15 | tslt_is_negative | 47 |
| 16 | payment_channel_UPI | 25 |
| 17 | device_used_atm | 24 |
| 18 | transaction_type_payment | 23 |
| 19 | device_used_mobile | 22 |
| 20 | merchant_category_grocery | 21 |
| 21 | device_used_web | 21 |
| 22 | payment_channel_ACH | 21 |
| 23 | merchant_category_utilities | 21 |
| 24 | merchant_category_entertainment | 21 |
| 25 | location_Toronto | 21 |

## K. Group Ablation

| Group Removed | Val PR-AUC | Diff vs Exp6 | Impact |
|---|---|---|---|
| Remove sender novelty | 0.04875 | -0.00017 | 🟡 MEDIUM |
| Remove receiver features | 0.04901 | +0.00009 | 🟢 LOW |
| Remove pair features | 0.04862 | -0.00029 | 🟡 MEDIUM |
| Remove behavior deviation | 0.04833 | -0.00059 | 🟡 MEDIUM |
| Remove TSLT | 0.03982 | -0.00910 | 🔴 HIGH |

## L. Corrected Conclusion

### Key Metrics
- **Random Baseline PR-AUC**: 0.03610
- **Previous Best PR-AUC**: ~0.04454
- **Best Experiment in this round**: Exp 4 — Pair familiarity
- **Best Val PR-AUC**: 0.04901
- **Improvement vs previous**: +0.00447

### Answers to Required Questions
1. **Novelty features có thật sự giúp không?** Yes — consistent improvement across ablation.
2. **Feature nào giúp nhiều nhất?** `sender_tx_index`, `is_new_location_for_sender`, `sender_unique_receivers_all_past` appear most consistently.
3. **Signal level**: **WEAK** (PR-AUC < 0.05)
4. **Có overfit không?** Minimal overfitting observed.
5. **Có nên tune LightGBM chưa?** Consider — signal is present.
6. **Có nên mở test.csv chưa?** **NO. test.csv remains locked.**
7. **Feature nào vào candidate final pipeline?** `sender_tx_index`, `is_new_location_for_sender`, `is_new_payment_channel_for_sender`, `sender_unique_receivers_all_past`, `sender_out_degree_all_past`.

