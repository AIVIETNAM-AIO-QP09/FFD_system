# GRAPH FEASIBILITY AUDIT REPORT — TRAIN/VALIDATION ONLY

## A. Objective
Determine if the dataset contains strong graph signals justifying Node2Vec or GNNs.
## B. Confirmation
`test.csv` is NOT used. Audit strictly on `train.csv`.

## C. Graph Construction Design
Time-aware rolling edges. Nodes: sender, receiver, device.

## D. Graph Feasibility Statistics (Train Inner Snapshot)

- **Edges**: 3,200,000
- **Unique Senders**: 874,391
- **Unique Receivers**: 874,419
- **Unique Devices**: 2,692,325
- **Sender Out-Degree**: Avg=3.66, Median=3.0, P99=9.0, Max=15
- **% Devices shared by multiple senders**: 16.75%
- **% Sender-Receiver pairs repeating**: 0.00%

*Conclusion*: Graph shows moderate connectivity and repeated structures.

## E. Non-target Graph Feature Diagnostics

| Feature | Null% | Zero% | Max | PR-AUC | Lift vs Rand |
|---|---|---|---|---|---|
| sender_out_degree_past | 0.0% | 1.9% | 17 | 0.03842 | 1.06x |
| sender_unique_receivers_past | 0.0% | 1.9% | 17 | 0.03842 | 1.06x |
| receiver_in_degree_past | 0.0% | 1.9% | 18 | 0.03595 | 1.00x |
| receiver_unique_senders_past | 0.0% | 1.9% | 18 | 0.03595 | 1.00x |
| device_sender_count_past | 0.0% | 67.0% | 6 | 0.03606 | 1.00x |
| sender_receiver_edge_count_past | 0.0% | 100.0% | 1 | 0.03610 | 1.00x |

## F. Past-label Graph Risk Feature Diagnostics

> [!IMPORTANT]
> **Mode B Assumption**: Past labels are perfectly available at time T with zero delay. This may overestimate performance.

| Feature | Null% | Zero% | Max | PR-AUC | Lift vs Rand |
|---|---|---|---|---|---|
| sender_past_fraud_neighbor_count_1hop | 0.0% | 76.1% | 5 | 0.03652 | 1.01x |
| device_past_fraud_sender_count | 0.0% | 97.2% | 3 | 0.03613 | 1.00x |

## G. Modeling Results

| Experiment | Features Added | Val PR-AUC | Prec@1% | Prec@5% | Block PR Range |
|---|---|---|---|---|---|
| Exp 0 | Baseline FS1 | 0.04882 | 4.86% | 4.89% | 0.00078 |
| Exp 1 | FS1 + Non-target Degree | 0.04888 | 5.15% | 5.00% | 0.00148 |
| Exp 2 | FS1 + Device Sharing | 0.04850 | 4.31% | 4.89% | 0.00090 |
| Exp 3 | FS1 + Pair Edges | 0.04882 | 4.86% | 4.89% | 0.00078 |
| Exp 4 | FS1 + All Non-target | 0.04912 | 5.49% | 5.07% | 0.00139 |
| Exp 5 | FS1 + Past-label Risk (Zero-delay) | 0.04878 | 4.84% | 4.92% | 0.00057 |

## H. Hard Subset Results

| Model | PR-AUC | Random Baseline | Lift |
|---|---|---|---|
| Exp 4 on Hard Subset | 0.04881 | 0.04401 | 1.109x |

## I. Leakage and Production Validity Review
- All features generated using strict temporal logic. No future leakage.
- Exp 1-4: Safe for production.
- Exp 5: Assumes zero-delay fraud labels. In reality, fraud chargebacks take weeks. This feature may not perform as well in real-time.

## J. Node2Vec / GNN Feasibility Decision

1. **Is graph structure dense enough?** The graph is extremely sparse. Shared devices and repeated sender-receiver pairs are very low.
2. **Do graph features improve validation PR-AUC?** Minimal to none. The PR-AUC lift from non-target graph features is negligible compared to FS1 baseline.
3. **Is improvement stable?** Yes, but the base signal is flat.
4. **Are label-based graph features production-safe?** No, zero-delay is unrealistic for this domain.
5. **Is Node2Vec justified?** No. Random walks on an overly sparse graph with no strong community structure will generate noise.
6. **Is GNN justified?** No. Message passing in a star-graph/sparse topology leads to over-smoothing without benefit.

## K. Final Recommendation

**Graph not useful for this dataset**

The dataset lacks the dense network topology (hubs, shared devices, tight clusters) required to justify the engineering overhead of Graph Representation Learning (Node2Vec/GNN). Stick with the tabular baseline.

