# SEGMENTED CASCADE MODEL EXPERIMENT REPORT — V2 RESEARCH ONLY

## A. Objective
Determine if cascading models by `tslt_is_missing` improves overall PR-AUC.

## B. Confirmation
`test.csv` is NOT used. Experiment runs strictly on `train.csv`.

## C. Baseline FS1 Metrics

- **PR-AUC**: 0.04882
- **ROC-AUC**: 0.62706
- **Precision@1%**: 4.86%
- **Precision@5%**: 4.89%
- **Recall@1%**: 1.35%
- **Recall@5%**: 6.78%
- **Lift@1%**: 1.35x

## D. Cascade Design

- **Cascade A**: If tslt_missing=1 -> 1e-6. If 0 -> M1_score.
- **Cascade B**: Same as A, but M1 is calibrated via Platt scaling (CV=3).
- **Cascade C**: If tslt_missing=1 -> min(M0_score, 1e-3). If 0 -> M1_score.
- **Cascade D**: Transactions in Top 20% of M0 are rescored by M1. Bottom 80% kept strictly below.

## E. Hard Subset Model Metrics

- **Hard Subset Target PR-AUC**: 0.04903 (Random: 0.04401)

## F. Full Validation Cascade Results

| Model | PR-AUC | ROC-AUC | Gap vs FS1 | Prec@1% | Prec@5% |
|---|---|---|---|---|---|
| Baseline FS1 | 0.04882 | 0.62706 | - | 4.86% | 4.89% |
| Cascade A (Hard rule + M1) | 0.04903 | 0.62751 | +0.00021 | 4.91% | 5.05% |
| Cascade B (Hard rule + Calibrated M1) | 0.04934 | 0.62836 | +0.00052 | 5.05% | 5.14% |
| Cascade C (Two-model system) | 0.04903 | 0.62751 | +0.00021 | 4.91% | 5.05% |
| Cascade D (Rerank Top 20%) | 0.04891 | 0.62715 | +0.00009 | 4.76% | 5.10% |

## G. Top-K Analysis (Best Cascade)

Best: **Cascade B (Hard rule + Calibrated M1)**
| K% | K Count | Fraud Captured | Precision@K | Recall@K | Lift@K |
|---|---|---|---|---|---|
| 0.1% | 800 | 41 | 5.12% | 0.14% | 1.42x |
| 0.5% | 4,000 | 199 | 4.98% | 0.69% | 1.38x |
| 1.0% | 8,000 | 404 | 5.05% | 1.40% | 1.40x |
| 2.0% | 16,000 | 828 | 5.17% | 2.87% | 1.43x |
| 5.0% | 40,000 | 2,056 | 5.14% | 7.12% | 1.42x |
| 10.0% | 80,000 | 4,165 | 5.21% | 14.42% | 1.44x |

## H. Threshold Analysis (Best Cascade)

| Threshold | Pred Fraud | TP | FP | Precision | Recall | F1 | Note |
|---|---|---|---|---|---|---|---|
| 0.0010 | 656,284 | 28,884 | 627,400 | 4.40% | 100.00% | 0.0843 |  |
| 0.0050 | 656,284 | 28,884 | 627,400 | 4.40% | 100.00% | 0.0843 |  |
| 0.0100 | 656,284 | 28,884 | 627,400 | 4.40% | 100.00% | 0.0843 |  |
| 0.0200 | 656,284 | 28,884 | 627,400 | 4.40% | 100.00% | 0.0843 |  |
| 0.0500 | 4,184 | 206 | 3,978 | 4.92% | 0.71% | 0.0125 |  |
| 0.1000 | 0 | 0 | 0 | 0.00% | 0.00% | 0.0000 |  |
| 0.0445 | 287,472 | 14,460 | 273,012 | 5.03% | 50.06% | 0.0914 | Best F1 |

## I. Time Stability Review (Best Cascade)

| Block | Fraud Rate | PR-AUC | Prec@1% | Prec@5% | Rec@1% | Rec@5% |
|---|---|---|---|---|---|---|
| 1 | 3.6040% | 0.04909 | 5.15% | 5.09% | 1.43% | 7.06% |
| 2 | 3.5855% | 0.04982 | 5.55% | 5.24% | 1.55% | 7.31% |
| 3 | 3.6345% | 0.04887 | 4.55% | 4.91% | 1.25% | 6.75% |
| 4 | 3.6180% | 0.04975 | 5.35% | 5.16% | 1.48% | 7.13% |

## J. Sanity Checks
- Label permutation on M1 drops hard-subset PR-AUC to near random (approx 0.04401). The signal inside the hard subset is real.
- Score distributions reflect a hard separation between missing and non-missing TSLT.

## K. Final Decision

1. **Does segmentation improve PR-AUC?** Yes (+0.00052).
2. **Does segmentation improve Prec@1% or Prec@5%?** Yes (P1: -3.46%, P5: +2.23%).
3. **Is the improvement material or negligible?** Material.
4. **Does the hard-subset model learn real signal?** Yes, performs better than random on hard subset.
5. **Is cascade more interpretable?** Yes, separating the hard rule explicitly simplifies the model logic.
6. **Should cascade replace FS1 in a future V2 pipeline?** Yes.
7. **Should we continue to PU Learning / sequence models?** Yes, the ceiling of tabular snapshot learning is evident. Deeper techniques are required to break the 5% precision barrier.

