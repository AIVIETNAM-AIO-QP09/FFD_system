# LM Studio Qwen3-8B Fraud Investigation Co-pilot Samples

A local Qwen3-8B model was used to generate candidate analyst wording. A strict validator-first layer replaced unsupported LLM claims with deterministic evidence-grounded text.

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
Transaction occurred at night, sender is using a new location for the first time, and amount via card deposit.

Suggested Analyst Next Step:
Verify if the sender's new location aligns with their account history. Confirm transaction context through additional communication channels. Check for any associated device or channel anomalies.

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
The transaction involves a new location, device type, payment channel, and transaction type, combined with a relatively high risk band score.

Suggested Analyst Next Step:
Verify the sender's identity through available authentication methods. If available, cross-check the sender's account activity against historical patterns. If available, confirm whether the UPI transfer is consistent with the sender's typical behavior. If available, review the device and location for any prior transaction history. If available, assess the payment channel's usual usage by the sender.

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
Transaction T1708346 was flagged due to the use of a new location, new device type, and new payment channel, combined with an unusual transaction time. These factors contributed to a low model risk score, prompting manual review.

Suggested Analyst Next Step:
Verify if the sender has any prior transaction history with the same location, device, or payment channel. Confirm the legitimacy of the transaction by contacting the sender directly. If available, check for any associated account activity or behavioral patterns that may contextualize the transaction.

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
Sender used new location, device type, payment channel, and transaction type. Transaction occurred at night. Amount was $1212.99 via card (deposit).

Suggested Analyst Next Step:
Verify sender's identity and intent. Confirm the transaction details with the sender. Check for any additional contextual factors that may explain the new elements. If available, review similar historical cases to assess patterns.

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
Transaction T2618028 was flagged due to the sender using a new location for the first time, the transaction occurring at night, and the withdrawal amount of $58.35 via card.

Suggested Analyst Next Step:
Verify if the sender's new location is associated with any known fraudulent activity. If available, check the device used for the transaction against any flagged devices. Review the sender's account history for any prior transactions from this location or at this time. Confirm whether the withdrawal amount aligns with typical spending patterns for the sender.

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
Transaction T3124730 was flagged due to the sender using a new location, new device type, and new transaction type. The transaction occurred at night, which may be less common for certain types of transactions.

Suggested Analyst Next Step:
Verify if the sender has any prior history with the recipient or account. Confirm the legitimacy of the transaction by checking the sender's account activity and communication channels used. If available, review similar historical cases to assess patterns.

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
Transaction T2327555 had a model risk score of 0.0000 and was assigned to the low-risk band. The sender used new location, device type, and transaction type, which are not indicative of fraud in this context. The transaction amount is small ($0.74), and it occurred via ACH, aligning with historical legitimate transactions. TSLT is missing, further reducing risk.

Suggested Analyst Next Step:
If available, verify the sender's identity and confirm the legitimacy of the new location, device type, and transaction type. Review any additional contextual data to ensure no anomalies are present.

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
Transaction T124856 has a model risk score of 0.0000 and is classified in the low-risk band. The sender is using a new location for the first time, which does not align with historical patterns. The transaction amount is small ($14.01), and it occurred via UPI, which is consistent with similar legitimate transactions. No evidence of fraud was identified.

Suggested Analyst Next Step:
If available, verify the sender's identity and confirm the transaction context. Review the new location for any prior activity or anomalies. Ensure the UPI channel is functioning normally and there are no reported issues with the transaction method.

Caveats:
1. The model score is 0.0000 and does not indicate model-based suspicion under the current scoring policy.
2. TSLT-missing behavior may reflect a data artifact and must be monitored for drift.
3. Similar historical cases are nearest-neighbor retrieval examples, not causal proof.

---

