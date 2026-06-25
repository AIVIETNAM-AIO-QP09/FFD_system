# LM Studio Qwen3-8B Fraud Investigation Co-pilot Samples

This report was generated using a local Qwen3-8B model through LM Studio. A strict post-generation validator was applied. Any LLM output that violated evidence-grounding rules was replaced by a deterministic fallback explanation. The LLM is used for analyst-facing summarization only and does not make fraud decisions.

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
The transaction was flagged due to the sender using a new location for the first time, occurring at night (4h), and involving a card deposit of $1063.13. The model assigned a risk score of 0.1336, placing it in the high-risk band.

Suggested Analyst Next Step:
Verify if the sender's new location is legitimate and confirm the transaction context. If available, check for any prior activity from this location or device. Review the card deposit details to ensure accuracy.

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
The transaction involves a new location, device type, payment channel, and transaction type. The amount is $65.21 via UPI (transfer). Similar historical cases include transactions via UPI with amounts ranging from $38.63 to $47.15.

Suggested Analyst Next Step:
Verify the sender's identity and confirm the legitimacy of the transaction. If available, check if the sender has a history of using similar channels, devices, or locations. Confirm whether the transaction type is typical for the sender. If available, review the device fingerprint and location data for consistency with known patterns.

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
The transaction was flagged due to the sender using a new location, new device type, and new payment channel for the first time. The transaction occurred at night (3h), and the amount is $237.64 via card (withdrawal).

Suggested Analyst Next Step:
If available, verify the sender's identity and confirm the legitimacy of the transaction. Check if the new location, device type, and payment channel are associated with any known legitimate activity. Review the transaction context to determine if it aligns with typical behavior for the sender.

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
The transaction was flagged due to the use of a new location, new device type, new payment channel, and new transaction type. The transaction occurred at night, which is not typical for this sender.

Suggested Analyst Next Step:
Verify if the sender has previously used any of these features. If available, check the historical behavior of similar transactions in the same context. Confirm the legitimacy of the transaction through additional verification steps with the sender.

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
The transaction was flagged due to the sender using a new location for the first time, occurring at night, and involving a card withdrawal. The model risk score is 0.0488, which falls within the medium risk band.

Suggested Analyst Next Step:
If available, verify the sender's identity and confirm the legitimacy of the transaction. Review any additional contextual information that may help determine the transaction's validity.

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
This transaction was flagged due to the presence of multiple new factors, including a new location, new device type, and new transaction type. The transaction also occurred at night.

Suggested Analyst Next Step:
If available, verify the sender's identity and confirm the legitimacy of the transaction. Review the sender's account history for any prior transactions from similar locations or devices. Check if the wire_transfer channel is typically used by the sender. Confirm whether the transaction aligns with the sender's usual behavior.

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
1. TSLT is missing; historically this group has near-zero fraud. This strongly lowers risk.
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
The transaction was not escalated due to the model's low risk score and the presence of features historically associated with legitimate activity. The sender is using new location, device type, and transaction type, which are present in similar historical cases labeled as legitimate. The amount is small, and the channel used (ACH) aligns with known legitimate transactions.

Suggested Analyst Next Step:
If available, verify if the sender's account has been previously associated with ACH transactions at night. If available, confirm whether the new location, device type, and transaction type are consistent with the sender's historical activity. If available, check for any additional contextual data that may further support the legitimacy of this transaction.

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
1. TSLT is missing; historically this group has near-zero fraud. This strongly lowers risk.
2. Sender is using a new location for the first time.
3. Transaction occurred at night (3h).
4. Amount: $14.01 via UPI (transfer)

Similar Historical Cases:
1. Dist: 2.68 | Label: Legitimate | Amount: $15.95 | Channel: card | Device: web
2. Dist: 3.04 | Label: Legitimate | Amount: $80.16 | Channel: wire_transfer | Device: mobile
3. Dist: 3.07 | Label: Fraud | Amount: $16.31 | Channel: UPI | Device: pos

Why This Was Not Escalated:
The transaction was not escalated due to the model risk score being 0.0000, which falls within the low-risk band. The evidence indicates that TSLT is missing, and this group historically has near-zero fraud, strongly lowering risk. The sender used a new location for the first time, and the transaction occurred at night (3h). The amount of $14.01 was transferred via UPI.

Suggested Analyst Next Step:
If available, verify whether the sender's new location is consistent with their historical activity. If available, confirm that the UPI transfer method and device used are typical for this sender. If available, check if any other transactions from this sender in the same time frame have similar characteristics.

Caveats:
1. The model score is 0.0000 and does not indicate model-based suspicion under the current scoring policy.
2. TSLT-missing behavior may reflect a data artifact and must be monitored for drift.
3. Similar historical cases are nearest-neighbor retrieval examples, not causal proof.


---

