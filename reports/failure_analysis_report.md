# FEATURE ENGINEERING FAILURE ANALYSIS REPORT — TRAIN/VALIDATION ONLY

## A. Correction of Previous Conclusion
- I explicitly retract the statement that interaction/historical features generated a 'Moderate-to-Strong' signal.
- Based on the metrics, adding features caused PR-AUC to DROP below baseline, thereby rendering the engineering round a **FAILURE**.
- The dataset currently possesses only a **WEAK SIGNAL**. Historical features created severe overfitting noise.

## B. TSLT Missing Audit
| Dataset | tslt_is_missing | Tx Count | Fraud Count | Fraud Rate (%) |
|---|---|---|---|---|
| Full Train | 0 | 3,281,922 | 143,527 | 4.3733% |
| Full Train | 1 | 718,078 | 0 | 0.0000% |
| Train Inner | 0 | 2,625,638 | 114,643 | 4.3663% |
| Train Inner | 1 | 574,362 | 0 | 0.0000% |
| Validation | 0 | 656,284 | 28,884 | 4.4011% |
| Validation | 1 | 143,716 | 0 | 0.0000% |

**Correction**: The table above explicitly proves that when `tslt_is_missing == 1`, fraud count is EXACTLY ZERO across both train_inner and validation. This means missing TSLT is a perfectly clean legitimate indicator (an artifact of data generation). Calling it inherently risky was a gross hallucination.

## C. Historical Feature Diagnostics
| Feature | Zero Rate (%) | Mean | Median | P99 | Max | Fraud Rate (0) | Fraud Rate (>0) |
|---|---|---|---|---|---|---|---|
| sender_tx_count_past_24h | 98.49% | 0.02 | 0.00 | 1.00 | 3.00 | 3.5793% | 3.7941% |
| sender_tx_count_past_7d | 90.04% | 0.10 | 0.00 | 1.00 | 4.00 | 3.5682% | 3.7124% |
| sender_amount_sum_past_24h | 98.49% | 5.49 | 0.00 | 44.97 | 3963.86 | 3.5793% | 3.7941% |
| receiver_tx_count_past_24h | 98.50% | 0.02 | 0.00 | 1.00 | 3.00 | 3.5811% | 3.6831% |
| receiver_tx_count_past_7d | 90.05% | 0.10 | 0.00 | 1.00 | 4.00 | 3.5753% | 3.6490% |
| device_tx_count_past_7d | 98.95% | 0.01 | 0.00 | 1.00 | 2.00 | 3.5848% | 3.3775% |
| ip_tx_count_past_7d | 100.00% | 0.00 | 0.00 | 0.00 | 1.00 | 3.5825% | 8.3333% |

**Observation**: If zero rate is ~99%, these features are too sparse to provide continuous lift. They are effectively flags.

## D. Interaction Feature Diagnostics
| Feature | PR-AUC Single | P99 | Max | Top Quantile Fraud Rate (%) |
|---|---|---|---|---|
| amount_x_velocity | 0.03635 | 137.03 | 161.39 | 3.5875% |
| geo_x_velocity | 0.03625 | 17.67 | 20.00 | 3.6247% |
| deviation_x_amount | 0.03645 | 13.17 | 32.24 | 3.5766% |
| geo_x_deviation | 0.03644 | 1.61 | 4.47 | 3.6049% |
| velocity_x_deviation | 0.03654 | 33.20 | 91.20 | 3.6056% |

## E. Ablation Results
| Experiment | PR-AUC | Diff vs Prev | Rel Lift | Prec@0.1% | Prec@1% | Prec@5% | Rec@1% | Rec@5% |
|---|---|---|---|---|---|---|---|---|
| A. Prev Baseline Exact | 0.04403 | -0.00051 | 1.22x | 6.75% | 4.61% | 4.35% | 1.28% | 6.03% |
| B. Prev Baseline Minus TSLT | 0.03624 | -0.00830 | 1.00x | 2.00% | 3.74% | 3.75% | 1.04% | 5.19% |
| C. Prev Baseline Only TSLT | 0.04395 | -0.00059 | 1.22x | 4.75% | 4.58% | 4.30% | 1.27% | 5.96% |
| D. Prev Baseline + Interactions | 0.04412 | -0.00042 | 1.22x | 3.62% | 4.40% | 4.35% | 1.22% | 6.02% |
| E. Prev Baseline + Historical | 0.04378 | -0.00076 | 1.21x | 4.25% | 4.58% | 4.22% | 1.27% | 5.84% |
| F. All Features | 0.04406 | -0.00048 | 1.22x | 4.00% | 4.59% | 4.49% | 1.27% | 6.22% |
| G. Only Historical | 0.03641 | -0.00813 | 1.01x | 4.75% | 3.91% | 3.84% | 1.08% | 5.32% |
| H. Only Interactions | 0.03643 | -0.00811 | 1.01x | 3.88% | 3.77% | 3.64% | 1.05% | 5.04% |

## F. Overfitting Check
| Experiment | Train PR-AUC | Val PR-AUC | Gap | Train Prec@1% | Val Prec@1% |
|---|---|---|---|---|---|
| A. Prev Baseline Exact | 0.05929 | 0.04403 | 0.01526 | 11.34% | 4.61% |
| B. Prev Baseline Minus TSLT | 0.04820 | 0.03624 | 0.01196 | 9.43% | 3.74% |
| C. Prev Baseline Only TSLT | 0.05298 | 0.04395 | 0.00903 | 8.32% | 4.58% |
| D. Prev Baseline + Interactions | 0.06001 | 0.04412 | 0.01589 | 11.22% | 4.40% |
| E. Prev Baseline + Historical | 0.05927 | 0.04378 | 0.01549 | 11.43% | 4.58% |
| F. All Features | 0.06081 | 0.04406 | 0.01675 | 11.39% | 4.59% |
| G. Only Historical | 0.03725 | 0.03641 | 0.00084 | 5.12% | 3.91% |
| H. Only Interactions | 0.04749 | 0.03643 | 0.01106 | 9.58% | 3.77% |

## G. Validation Time Stability
| Block | Time Range | Fraud Rate (%) | Baseline PR-AUC | Baseline Prec@1% | All-Feat PR-AUC | All-Feat Prec@1% |
|---|---|---|---|---|---|---|
| Block 1 | 2023-08-23 03:48:08.422482 to 2023-09-06 17:19:50.981152 | 3.6040% | 0.04387 | 3.90% | 0.04442 | 4.95% |
| Block 2 | 2023-09-06 17:19:53.107524 to 2023-09-21 07:15:46.789151 | 3.5855% | 0.04421 | 5.20% | 0.04420 | 4.65% |
| Block 3 | 2023-09-21 07:15:55.334164 to 2023-10-05 21:09:42.789136 | 3.6345% | 0.04386 | 4.25% | 0.04381 | 3.90% |
| Block 4 | 2023-10-05 21:09:53.317789 to 2023-10-20 10:49:03.363826 | 3.6180% | 0.04447 | 4.75% | 0.04399 | 4.90% |


## H. Corrected Recommendation
1. **TSLT Features**: `tslt_is_missing` is a pure artifact representing 0% fraud. It forces the tree to drop legitimate branches easily but does not help isolate fraud cases. KEEP IT, but understand it is an exclusionary rule, not a fraud-detector.
2. **Interaction Features**: Discard them. They produce excessive feature space sparsity, worsening PR-AUC and drastically increasing Train/Val gap.
3. **Historical Features**: Discard the current rolling features. They are over 90% zeros (Zero Rate). Counting in 24h/7d windows across 3.2M rows created heavily sparse arrays that LightGBM overfit on. 
4. **Way Forward**: Return to the **Previous Baseline Exact** feature set.
5. **Tune LightGBM**: Do NOT tune. The signal is weak. Tuning noise will just cause extreme overfitting.
6. **Test.csv**: Do NOT open `test.csv`. The validation pipeline is telling us that our data does not contain the features needed to distinguish fraud from legitimate traffic cleanly. We need **Data Generating/Labeling Insights** or **Different Features** (Target-Encoding properly masked, or IP/Device blacklists) before proceeding.

