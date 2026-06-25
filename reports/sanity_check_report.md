# SANITY CHECK REPORT — Feature Discovery Pre-Validation

## A. Objective
Verify whether dataset contains any real predictive signal before committing to expensive feature engineering.
Only `train.csv` used. `test.csv` untouched.

## B. Current Evidence Summary — Fraud Rate Uniformity
**Overall fraud rate**: 0.03583

### transaction_type
Max fraud rate: 0.03608 (1.007x) | Min: 0.03568 (0.996x)

| Category | Count | Fraud Rate | Lift |
|---|---|---|---|
| deposit | 799,565 | 3.5686% | 0.9961x |
| payment | 800,017 | 3.5680% | 0.9959x |
| transfer | 801,133 | 3.6079% | 1.0071x |
| withdrawal | 799,285 | 3.5858% | 1.0009x |

### merchant_category
Max fraud rate: 0.03622 (1.011x) | Min: 0.03548 (0.990x)

| Category | Count | Fraud Rate | Lift |
|---|---|---|---|
| entertainment | 399,995 | 3.5720% | 0.9971x |
| grocery | 399,665 | 3.6218% | 1.0109x |
| online | 399,621 | 3.5781% | 0.9988x |
| other | 399,771 | 3.6108% | 1.0079x |
| restaurant | 400,057 | 3.5485% | 0.9905x |
| retail | 400,153 | 3.5674% | 0.9958x |
| travel | 400,979 | 3.5830% | 1.0001x |
| utilities | 399,759 | 3.5792% | 0.9990x |

### payment_channel
Max fraud rate: 0.03590 (1.002x) | Min: 0.03576 (0.998x)

| Category | Count | Fraud Rate | Lift |
|---|---|---|---|
| ACH | 799,266 | 3.5785% | 0.9989x |
| UPI | 799,449 | 3.5757% | 0.9981x |
| card | 799,802 | 3.5896% | 1.0020x |
| wire_transfer | 801,483 | 3.5865% | 1.0011x |

### device_used
Max fraud rate: 0.03587 (1.001x) | Min: 0.03579 (0.999x)

| Category | Count | Fraud Rate | Lift |
|---|---|---|---|
| atm | 799,780 | 3.5871% | 1.0013x |
| mobile | 801,043 | 3.5788% | 0.9990x |
| pos | 799,395 | 3.5846% | 1.0006x |
| web | 799,782 | 3.5799% | 0.9992x |

### location
Max fraud rate: 0.03623 (1.011x) | Min: 0.03555 (0.992x)

| Category | Count | Fraud Rate | Lift |
|---|---|---|---|
| Berlin | 400,580 | 3.5758% | 0.9981x |
| Dubai | 399,178 | 3.5551% | 0.9923x |
| London | 399,620 | 3.5669% | 0.9956x |
| New York | 399,911 | 3.5808% | 0.9995x |
| Singapore | 399,852 | 3.5951% | 1.0035x |
| Sydney | 400,399 | 3.5942% | 1.0032x |
| Tokyo | 401,003 | 3.5703% | 0.9966x |
| Toronto | 399,457 | 3.6227% | 1.0112x |

**Verdict**: If max lift ≈ min lift ≈ 1.0x across all categories, labels are uniformly distributed (synthetic random data).


## G. New Feature Signal Diagnostics (Single-Feature PR-AUC)
Validation fraud rate (random baseline): **0.03610**

| Feature | PR-AUC | Rel Lift | Direction | Assessment |
|---|---|---|---|---|
| amount | 0.03636 | 1.007x | neg | ❌ Noise |
| log_amount | 0.03636 | 1.007x | neg | ❌ Noise |
| spending_deviation_score | 0.03652 | 1.011x | pos | ❌ Noise |
| velocity_score | 0.03624 | 1.004x | neg | ❌ Noise |
| geo_anomaly_score | 0.03617 | 1.002x | neg | ❌ Noise |
| hour | 0.03618 | 1.002x | pos | ❌ Noise |
| day_of_week | 0.03623 | 1.003x | pos | ❌ Noise |
| month | 0.03614 | 1.001x | neg | ❌ Noise |
| is_weekend | 0.03618 | 1.002x | pos | ❌ Noise |
| is_night | 0.03616 | 1.002x | neg | ❌ Noise |
| tslt_abs | 0.03938 | 1.091x | neg | ⚠️ Marginal |
| tslt_is_missing | 0.04401 | 1.219x | neg | ✅ Signal |
| tslt_is_negative | 0.03832 | 1.061x | pos | ⚠️ Marginal |
| transaction_type (mean-encoded) | 0.03640 | 1.008x | pos | ❌ Noise |
| merchant_category (mean-encoded) | 0.03610 | 1.000x | neg | ❌ Noise |
| payment_channel (mean-encoded) | 0.03615 | 1.001x | neg | ❌ Noise |
| device_used (mean-encoded) | 0.03662 | 1.014x | pos | ❌ Noise |
| location (mean-encoded) | 0.03643 | 1.009x | neg | ❌ Noise |

## K. Label Permutation Sanity Check
| Label Type | PR-AUC | ROC-AUC | Prec@1% | Rel Lift |
|---|---|---|---|---|
| Real Labels | 0.04403 | 0.59306 | 4.61% | 1.219x |
| Shuffled Labels | 0.03636 | 0.50021 | 3.61% | 1.007x |

**Gap (Real vs Shuffle)**: +0.00767 (+21.1%)

**Verdict**: 🟢 **SIGNAL EXISTS**: Real labels significantly outperform shuffle. Dataset contains learnable patterns.


## L. Final Recommendation (After Sanity Check)

Based on evidence collected:

**Fraud Rate Uniformity**: If all categories have lift ≈ 1.0x, categorical features carry no signal.

**Score Distributions**:
- `velocity_score`: Perfectly uniform 1–20 distribution → generated randomly
- `spending_deviation_score`: Standard Normal → generated randomly
- `geo_anomaly_score`: Uniform [0,1] → generated randomly

**Device/IP Cardinality**:
- `ip_address`: ~4M unique on 4M rows → 1 IP per transaction → zero historical signal possible
- `device_hash`: 3.2M unique → too sparse for historical features

**Label Permutation Gap**: See section K above.

If gap < 5%: **STOP MODELING**. Proceed with feature discovery only as academic exercise.
If gap 5–20%: Weak signal. Feature engineering may help marginally.
If gap > 20%: Real signal. Aggressive feature engineering warranted.

