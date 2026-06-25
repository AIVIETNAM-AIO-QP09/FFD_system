# LM Studio Qwen3-8B Fraud Investigation Co-pilot Samples

A local Qwen3-8B model was integrated through LM Studio to generate candidate analyst wording. Because unsupported risk language can appear in LLM outputs, the final analyst-facing report uses a validator-first deterministic guardrail layer. The LLM does not make fraud decisions.

Fraud Investigation Brief

Transaction ID: T2174172
Model Risk Score: 0.1336
Risk Band: High
Recommended Action: Escalate to senior analyst

Key Risk Factors:
1. Sender is using a new location for the first time.
2. Transaction occurred at night (4h).
3. Amount: $1063.13 via card (deposit)

Similar Historical Cases:
1. Dist: 2.81 | Label: Fraud | Amount: $1312.66 | Channel: UPI | Device: web
2. Dist: 3.37 | Label: Legitimate | Amount: $1738.27 | Channel: card | Device: web
3. Dist: 3.47 | Label: Legitimate | Amount: $1247.46 | Channel: UPI | Device: mobile

Why This Was Flagged:
This transaction was selected for review based on its risk band and the listed structured evidence. Retrieved neighbors include historical labels for context only and are nearest-neighbor examples, not causal proof of fraud.

Suggested Analyst Next Step:
Escalate to a senior analyst for manual review. If available, verify sender identity and compare the transaction with recent legitimate activity from the same sender. Do not block or freeze funds based only on this report.

Caveats:
1. The underlying ML model is a weak ranking signal and high rank does not guarantee fraud.
2. Similar historical cases are nearest-neighbor retrieval examples, not causal proof.
3. The listed features are evidence for review routing, not definitive fraud indicators.

---

Fraud Investigation Brief

Transaction ID: T4534855
Model Risk Score: 0.0806
Risk Band: High
Recommended Action: Escalate to senior analyst

Key Risk Factors:
1. Sender is using a new location for the first time.
2. Sender is using a new device type for the first time.
3. Sender is using a new payment channel.
4. Sender is using a new transaction type.
5. Amount: $65.21 via UPI (transfer)

Similar Historical Cases:
1. Dist: 2.92 | Label: Fraud | Amount: $47.15 | Channel: UPI | Device: web
2. Dist: 3.17 | Label: Fraud | Amount: $38.63 | Channel: UPI | Device: atm
3. Dist: 3.23 | Label: Legitimate | Amount: $26.59 | Channel: card | Device: atm

Why This Was Flagged:
This transaction was selected for review based on its risk band and the listed structured evidence. Retrieved neighbors include historical labels for context only and are nearest-neighbor examples, not causal proof of fraud.

Suggested Analyst Next Step:
Escalate to a senior analyst for manual review. If available, verify sender identity and compare the transaction with recent legitimate activity from the same sender. Do not block or freeze funds based only on this report.

Caveats:
1. The underlying ML model is a weak ranking signal and high rank does not guarantee fraud.
2. Similar historical cases are nearest-neighbor retrieval examples, not causal proof.
3. The listed features are evidence for review routing, not definitive fraud indicators.

---

Fraud Investigation Brief

Transaction ID: T1708346
Model Risk Score: 0.0539
Risk Band: Review
Recommended Action: Manual review

Key Risk Factors:
1. Sender is using a new location for the first time.
2. Sender is using a new device type for the first time.
3. Sender is using a new payment channel.
4. Transaction occurred at night (3h).
5. Amount: $237.64 via card (withdrawal)

Similar Historical Cases:
1. Dist: 3.07 | Label: Legitimate | Amount: $291.74 | Channel: wire_transfer | Device: atm
2. Dist: 3.48 | Label: Fraud | Amount: $212.24 | Channel: ACH | Device: web
3. Dist: 3.61 | Label: Fraud | Amount: $74.81 | Channel: ACH | Device: web

Why This Was Flagged:
This transaction was selected for review based on its risk band and the listed structured evidence. Retrieved neighbors include historical labels for context only and are nearest-neighbor examples, not causal proof of fraud.

Suggested Analyst Next Step:
Perform manual review. If available, check whether the listed new attributes are consistent with recent legitimate activity from the same sender.

Caveats:
1. The underlying ML model is a weak ranking signal and high rank does not guarantee fraud.
2. Similar historical cases are nearest-neighbor retrieval examples, not causal proof.
3. The listed features are evidence for review routing, not definitive fraud indicators.

---

Fraud Investigation Brief

Transaction ID: T1888153
Model Risk Score: 0.0587
Risk Band: Review
Recommended Action: Manual review

Key Risk Factors:
1. Sender is using a new location for the first time.
2. Sender is using a new device type for the first time.
3. Sender is using a new payment channel.
4. Sender is using a new transaction type.
5. Transaction occurred at night (3h).
6. Amount: $1212.99 via card (deposit)

Similar Historical Cases:
1. Dist: 3.79 | Label: Fraud | Amount: $266.67 | Channel: ACH | Device: mobile
2. Dist: 3.95 | Label: Fraud | Amount: $498.98 | Channel: wire_transfer | Device: mobile
3. Dist: 4.14 | Label: Legitimate | Amount: $823.17 | Channel: card | Device: atm

Why This Was Flagged:
This transaction was selected for review based on its risk band and the listed structured evidence. Retrieved neighbors include historical labels for context only and are nearest-neighbor examples, not causal proof of fraud.

Suggested Analyst Next Step:
Perform manual review. If available, check whether the listed new attributes are consistent with recent legitimate activity from the same sender.

Caveats:
1. The underlying ML model is a weak ranking signal and high rank does not guarantee fraud.
2. Similar historical cases are nearest-neighbor retrieval examples, not causal proof.
3. The listed features are evidence for review routing, not definitive fraud indicators.

---

Fraud Investigation Brief

Transaction ID: T2618028
Model Risk Score: 0.0488
Risk Band: Medium
Recommended Action: Monitor

Key Risk Factors:
1. Sender is using a new location for the first time.
2. Transaction occurred at night (3h).
3. Amount: $58.35 via card (withdrawal)

Similar Historical Cases:
1. Dist: 3.20 | Label: Legitimate | Amount: $187.76 | Channel: card | Device: atm
2. Dist: 3.34 | Label: Fraud | Amount: $37.41 | Channel: card | Device: atm
3. Dist: 3.44 | Label: Fraud | Amount: $425.87 | Channel: ACH | Device: atm

Why This Was Flagged:
This transaction was selected for review based on its risk band and the listed structured evidence. Retrieved neighbors include historical labels for context only and are nearest-neighbor examples, not causal proof of fraud.

Suggested Analyst Next Step:
Monitor under current policy. If available, review recent sender activity before taking any additional action.

Caveats:
1. The underlying ML model is a weak ranking signal and high rank does not guarantee fraud.
2. Similar historical cases are nearest-neighbor retrieval examples, not causal proof.
3. The listed features are evidence for review routing, not definitive fraud indicators.

---

Fraud Investigation Brief

Transaction ID: T3124730
Model Risk Score: 0.0524
Risk Band: Medium
Recommended Action: Monitor

Key Risk Factors:
1. Sender is using a new location for the first time.
2. Sender is using a new device type for the first time.
3. Sender is using a new transaction type.
4. Transaction occurred at night (4h).
5. Amount: $1415.50 via wire_transfer (deposit)

Similar Historical Cases:
1. Dist: 2.70 | Label: Legitimate | Amount: $1115.85 | Channel: wire_transfer | Device: web
2. Dist: 3.63 | Label: Fraud | Amount: $760.90 | Channel: ACH | Device: mobile
3. Dist: 3.68 | Label: Legitimate | Amount: $1686.71 | Channel: UPI | Device: pos

Why This Was Flagged:
This transaction was selected for review based on its risk band and the listed structured evidence. Retrieved neighbors include historical labels for context only and are nearest-neighbor examples, not causal proof of fraud.

Suggested Analyst Next Step:
Monitor under current policy. If available, review recent sender activity before taking any additional action.

Caveats:
1. The underlying ML model is a weak ranking signal and high rank does not guarantee fraud.
2. Similar historical cases are nearest-neighbor retrieval examples, not causal proof.
3. The listed features are evidence for review routing, not definitive fraud indicators.

---

Fraud Investigation Brief

Transaction ID: T2327555
Model Risk Score: 0.0000
Risk Band: Low
Recommended Action: Approve under current policy

Key Risk Factors:
1. TSLT is missing; historically this group has near-zero fraud. This is treated as a low-risk structural bucket under the current model.
2. Sender is using a new location for the first time.
3. Sender is using a new device type for the first time.
4. Sender is using a new transaction type.
5. Transaction occurred at night (3h).
6. Amount: $0.74 via ACH (payment)

Similar Historical Cases:
1. Dist: 3.42 | Label: Legitimate | Amount: $153.56 | Channel: ACH | Device: atm
2. Dist: 3.53 | Label: Fraud | Amount: $1.23 | Channel: ACH | Device: mobile
3. Dist: 3.66 | Label: Legitimate | Amount: $29.86 | Channel: ACH | Device: atm

Why This Was Not Escalated:
This transaction was not escalated because it falls into the TSLT-missing structural bucket, which historically showed near-zero fraud in the training/validation data. This dependency may reflect a data artifact and should be monitored for drift.

Suggested Analyst Next Step:
No immediate review required under current policy. Monitor TSLT-missing rate and fraud rate over time.

Caveats:
1. The model score is 0.0000 and does not indicate model-based suspicion under the current scoring policy.
2. TSLT-missing behavior may reflect a data artifact and must be monitored for drift.
3. Similar historical cases are nearest-neighbor retrieval examples, not causal proof.

---

Fraud Investigation Brief

Transaction ID: T124856
Model Risk Score: 0.0000
Risk Band: Low
Recommended Action: Approve under current policy

Key Risk Factors:
1. TSLT is missing; historically this group has near-zero fraud. This is treated as a low-risk structural bucket under the current model.
2. Sender is using a new location for the first time.
3. Transaction occurred at night (3h).
4. Amount: $14.01 via UPI (transfer)

Similar Historical Cases:
1. Dist: 2.68 | Label: Legitimate | Amount: $15.95 | Channel: card | Device: web
2. Dist: 3.04 | Label: Legitimate | Amount: $80.16 | Channel: wire_transfer | Device: mobile
3. Dist: 3.07 | Label: Fraud | Amount: $16.31 | Channel: UPI | Device: pos

Why This Was Not Escalated:
This transaction was not escalated because it falls into the TSLT-missing structural bucket, which historically showed near-zero fraud in the training/validation data. This dependency may reflect a data artifact and should be monitored for drift.

Suggested Analyst Next Step:
No immediate review required under current policy. Monitor TSLT-missing rate and fraud rate over time.

Caveats:
1. The model score is 0.0000 and does not indicate model-based suspicion under the current scoring policy.
2. TSLT-missing behavior may reflect a data artifact and must be monitored for drift.
3. Similar historical cases are nearest-neighbor retrieval examples, not causal proof.

---

## Productization Appendix — LM Studio Qwen3-8B Fraud Review Co-pilot

### 1. Objective
The goal is to convert the weak but validated fraud-ranking model into an analyst-facing review assistant. The LLM is not used as a fraud decision-maker.

### 2. Architecture
Raw transaction
→ Feature engineering
→ Cascade/LightGBM risk score
→ Evidence builder
→ Similar-case retriever
→ Qwen3-8B candidate wording through LM Studio
→ Validator-first deterministic guardrail layer
→ Analyst-facing fraud investigation brief

### 3. Local LLM Setup
* Runtime: LM Studio local server
* Model: Qwen3-8B GGUF Q4_K_M
* Endpoint: `http://127.0.0.1:1234/v1`
* Model identifier: `qwen/qwen3-8b:2`
* Temperature: 0.1
* Purpose: generate candidate analyst wording

### 4. Guardrail Design
* Qwen3 output is treated as candidate wording only.
* Unsupported risk language is blocked.
* Final analyst-facing text uses deterministic safe templates when needed.
* The LLM does not make fraud decisions.
* Similar cases are nearest-neighbor examples, not proof.
* No automatic blocking, fund freezing, or account suspension.

### 5. Final Safe Report Behavior
The final output sections are deterministically structured:
* Transaction ID
* Model Risk Score
* Risk Band
* Recommended Action
* Key Risk Factors
* Similar Historical Cases
* Why This Was Flagged / Why This Was Not Escalated
* Suggested Analyst Next Step
* Caveats

### 6. Limitations
* Underlying model remains weak.
* LLM does not improve PR-AUC or Precision@K.
* Current RAG uses structured nearest neighbors, not real case notes.
* Dataset lacks analyst comments, investigation notes, device intelligence, and external reputation data.
* TSLT-missing low-risk behavior may be a dataset artifact.

### 7. Final Positioning
This extension improves explainability, analyst workflow, and product presentation. It does not make the fraud detector more accurate and must not be used as an autonomous decision system.

### 8. Final Sample Report Summary
Sample briefs were successfully generated for:
* 2 High cases
* 2 Review cases
* 2 Medium cases
* 2 Low cases

The final guarded output passed the project safety criteria.

