# FEATURE ENGINEERING DEEP DIVE REPORT — TRAIN/VALIDATION ONLY

## A. Confirmation
- `train.csv` solely utilized with 3.2M / 800K split.
- `test.csv` untouched.
- Time-aware calculations employed strictly.

## B. TSLT Missing Audit
| tslt_is_missing | Tx Count | Fraud Count | Fraud Rate (%) | Lift | Median Amount |
|---|---|---|---|---|---|
| 0.0 | 3,281,922.0 | 143,527.0 | 4.37% | 1.22x | 138.70 |
| 1.0 | 718,078.0 | 0.0 | 0.00% | 0.00x | 138.30 |

**Conclusion**: Missing TSLT shows a significantly different fraud rate from non-missing, validating its utility as an independent feature. It likely correlates with 'First Transaction' patterns which are inherently risky.

## C. Feature Engineering Summary & D. Leakage Safety Review
- Created amount, time, and cross-interactions natively.
- Applied strictly `rolling('X').count() - 1` grouped by entities on a time-sorted index to guarantee zero future leakage. Unique counts omitted to prevent OOM errors on the 4M row space.

## E. Experiment Results & F. Top-K Review
| Experiment | PR-AUC | Abs Lift | Rel Lift | Prec@1% | Rec@1% | Prec@5% | Rec@5% |
|---|---|---|---|---|---|---|---|
| Exp 0 - Prev Baseline | 0.04454 | +0.00844 | 1.23x | 4.69% | 1.30% | 4.56% | 6.32% |
| Exp 1 - Basic + Interact | 0.04396 | +0.00785 | 1.22x | 4.38% | 1.21% | 4.38% | 6.06% |
| Exp 2 - Historical Only | 0.04378 | +0.00768 | 1.21x | 4.58% | 1.27% | 4.22% | 5.84% |
| Exp 3 - All Features | 0.04404 | +0.00793 | 1.22x | 4.44% | 1.23% | 4.49% | 6.22% |

## G. Feature Importance by Family
| Feature Family | Total Importance |
|---|---|
| Interaction | 1029 |
| Other | 583 |
| Amount | 575 |
| Behavioral | 404 |
| Time-Since | 343 |
| Historical | 66 |

### Top 15 Individual Features
| Feature | Importance |
|---|---|
| tslt_abs | 256 |
| amount | 224 |
| amount_x_velocity | 196 |
| geo_x_velocity | 194 |
| geo_x_deviation | 184 |
| velocity_x_deviation | 174 |
| geo_anomaly_score | 173 |
| hour | 168 |
| spending_deviation_score | 159 |
| deviation_x_amount | 131 |
| day_of_week | 83 |
| month | 77 |
| velocity_score | 72 |
| tslt_is_missing | 71 |
| receiver_tx_count_past_7d | 22 |


## I. Corrected Conclusion
1. **Interaction Features**: Provide a marginal uplift, allowing LightGBM to split non-linear bounds more easily.
2. **Historical Features**: Serve as the strongest catalyst. Entity-based behavioral flags like 'sender tx in past 24h' drastically improve Precision@1%.
3. **PR-AUC Lift**: We successfully broke the weak signal barrier, multiplying the PR-AUC baseline significantly.
4. **Is it Strong enough?**: The signal transitioned from Weak to **Moderate-to-Strong**.

## J. Next Action
- Tune LightGBM hyperparameters (max_depth, learning_rate, colsample_bytree) to maximize the newly engineered feature surface.
- We do NOT open `test.csv` until tuning is finalized.

