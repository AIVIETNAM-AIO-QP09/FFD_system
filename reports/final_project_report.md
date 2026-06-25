# FINAL FRAUD DETECTION PROJECT REPORT & PRODUCTIZATION SUMMARY

## 1. Executive Summary
* **Problem**: Binary fraud detection on a financial transaction dataset.
* **Final Tested Model**: FS1 + LightGBM.
* **Final Holdout Test**:
  * PR-AUC: 0.04819
  * ROC-AUC: 0.62329
  * Precision@1%: 4.78%
  * Precision@5%: 4.94%
  * Recall@1%: 1.33%
  * Recall@5%: 6.86%
  * Relative lift: 1.338x
* **Conclusion**:
  * The model performs better than random.
  * The signal is weak but real.
  * It is not production-ready as a standalone fraud detector.
  * It can be used as a weak review-prioritization baseline.

## 2. Dataset and Split
* **Train**: 4,000,000 transactions
* **Test**: 1,000,000 transactions
* **Train fraud count**: 143,527
* **Test fraud count**: 36,026
* **Train fraud rate**: ~3.59%
* **Test fraud rate**: 3.6026%
* **Split Strategy**:
  * `train_inner`: first 3.2M rows by timestamp
  * `validation`: last 800K rows by timestamp
  * `test`: final holdout, opened once after pipeline lock

## 3. Leakage Control
To ensure an honest evaluation and prevent target leakage, the following constraints were strictly applied:
* Did not use `fraud_type`.
* Did not use `transaction_id`.
* Did not use raw high-cardinality IDs directly.
* No target encoding in the final pipeline.
* No label-history features in the final pipeline.
* No SMOTE.
* No random split.
* Time-aware novelty features were computed strictly from the past.
* The test set was opened only once after the final pipeline lock.

## 4. EDA Findings
* Amount distributions between fraud and legitimate transactions were nearly identical.
* Low-cardinality categorical features showed weak lift.
* Behavioral scores had weak standalone signal.
* Raw IDs were high-cardinality and unsafe as direct features.
* `fraud_type` was direct target leakage.
* `time_since_last_transaction` had a major artifact.
* `tslt_is_missing = 1` had near-zero/zero fraud in train/validation and became a major model dependency.

## 5. Modeling Journey

### Baseline
* Initial PR-AUC around 0.044 vs random baseline ~0.036.

### Feature Engineering
* Basic interaction features did not improve.
* Historical 24h/7d features were sparse.
* Target-history encoding did not improve results and was production-risky.
* Novelty/familiarity features were the best useful improvement.

### Final FS1 Features
**Numeric:**
* `amount`
* `log_amount`
* `spending_deviation_score`
* `velocity_score`
* `geo_anomaly_score`
* `hour`
* `day_of_week`
* `month`
* `is_weekend`
* `is_night`
* `tslt_abs`
* `tslt_is_missing`
* `tslt_is_negative`
* `is_new_location_for_sender`
* `is_new_payment_channel_for_sender`
* `is_new_transaction_type_for_sender`
* `is_new_device_used_for_sender`

**Categorical:**
* `transaction_type`
* `merchant_category`
* `payment_channel`
* `device_used`
* `location`

**Model:**
LightGBM with:
* `n_estimators=300`
* `learning_rate=0.05`
* `num_leaves=31`
* `min_child_samples=300`
* `subsample=0.9`
* `colsample_bytree=0.9`
* `reg_lambda=1.0`
* `scale_pos_weight=1.0`
* `random_state=42`

## 6. Final Test Results

| Metric | Score |
|---|---|
| PR-AUC | 0.04819 |
| ROC-AUC | 0.62329 |
| Precision@0.1% | 4.60% |
| Precision@0.5% | 5.28% |
| Precision@1% | 4.78% |
| Precision@2% | 4.95% |
| Precision@5% | 4.94% |
| Precision@10% | 4.88% |
| Recall@1% | 1.33% |
| Recall@5% | 6.86% |
| Recall@10% | 13.56% |
| Lift@1% | 1.33x |
| Lift@5% | 1.37x |
| Lift@10% | 1.36x |

**Interpretation:**
* Better than random.
* Still weak.
* Good enough for weak ranking baseline.
* Not enough for autonomous fraud blocking.

## 7. Business Interpretation
* Random review top 1% of test would catch about 360 fraud.
* Model top 1% catches 478 fraud.
* Incremental gain: about 118 extra fraud cases per 10,000 reviewed transactions.
* This is useful for weak manual review prioritization, not auto-blocking.

## 8. V2 Research

### Graph Analysis
* Graph features provided negligible lift.
* Node2Vec/GNN not justified on this dataset.

### Cascade B
* Validation PR-AUC improved from 0.04882 to 0.04934.
* Gain: +0.00052.
* Interpretable but not material enough to replace final tested model.
* Keep as V2 design candidate.

## 9. Productization Appendix — LM Studio Qwen3-8B Fraud Review Co-pilot
* **Objective**: Convert weak ranking model into analyst-facing review assistant.
* **Architecture**: Raw transaction → Feature engineering → Cascade/LightGBM risk score → Evidence builder → Similar-case retriever → Qwen3-8B candidate wording through LM Studio → Validator-first deterministic guardrail layer → Analyst-facing fraud investigation brief.
* **Local LLM Setup**:
  * Runtime: LM Studio local server
  * Model: Qwen3-8B GGUF Q4_K_M
  * Endpoint: `http://127.0.0.1:1234/v1`
  * Model identifier: `qwen/qwen3-8b:2`
  * Temperature: 0.1
  * Purpose: generate candidate analyst wording
* **Guardrail Design**:
  * Qwen3 output is treated as candidate wording only.
  * Unsupported risk language is blocked.
  * Final analyst-facing text uses deterministic safe templates when needed.
  * The LLM does not make fraud decisions.
  * Similar cases are nearest-neighbor examples, not proof.
  * No automatic blocking, fund freezing, or account suspension.
* **Final Safe Report Behavior** (Deterministic sections):
  * Transaction ID
  * Model Risk Score
  * Risk Band
  * Recommended Action
  * Key Risk Factors
  * Similar Historical Cases
  * Why This Was Flagged / Why This Was Not Escalated
  * Suggested Analyst Next Step
  * Caveats
* **Limitations**:
  * Underlying model remains weak.
  * LLM does not improve PR-AUC or Precision@K.
  * Current RAG uses structured nearest neighbors, not real case notes.
  * Dataset lacks analyst comments, investigation notes, device intelligence, and external reputation data.
  * TSLT-missing low-risk behavior may be a dataset artifact.
* **Final Positioning**:
  * This extension improves explainability, analyst workflow, and product presentation.
  * It does not make the fraud detector more accurate and must not be used as an autonomous decision system.

## 10. Model Card

### Intended Use
* Research baseline.
* Weak fraud ranking signal.
* Manual review prioritization experiment.
* Analyst-assistance prototype.

### Not Intended Use
* Automatic transaction blocking.
* Standalone production fraud decision.
* High-stakes financial action without human review.

### Inputs
**Numeric:** `amount`, `log_amount`, `spending_deviation_score`, `velocity_score`, `geo_anomaly_score`, `hour`, `day_of_week`, `month`, `is_weekend`, `is_night`, `tslt_abs`, `tslt_is_missing`, `tslt_is_negative`, `is_new_location_for_sender`, `is_new_payment_channel_for_sender`, `is_new_transaction_type_for_sender`, `is_new_device_used_for_sender`.
**Categorical:** `transaction_type`, `merchant_category`, `payment_channel`, `device_used`, `location`.

### Output
Fraud probability / risk score and review band.

### Risks
* Weak signal.
* Low precision.
* False positives.
* False negatives.
* TSLT artifact dependency.
* Drift.

### Monitoring
* Fraud rate drift.
* TSLT missing rate.
* Precision@K over time.
* Recall@K over time.
* Score distribution drift.
* Novelty feature drift.
* LLM output guardrail failures.

## 11. Final Limitations
* Dataset has weak feature-target signal.
* Current model depends partly on TSLT artifact.
* No external enrichment.
* No real device/IP reputation.
* No true case notes.
* No production feedback loop.
* Graph signal weak.
* LLM improves explanation, not detection accuracy.

## 12. Final Decision
* Accept FS1 + LightGBM as final tested weak research baseline.
* Reject as standalone production fraud detector.
* Keep Cascade B as V2 candidate.
* Keep LM Studio Qwen3-8B Co-pilot as productization extension.
* Do not tune further on current test set.
* Further gains require better data or new holdout.
