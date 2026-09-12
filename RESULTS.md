# Full Fraud Analysis Results

These results were computed on the full **590,540-row** public model-ready IEEE-CIS feature table derived from the original Kaggle/Vesta data. The repository still preserves the raw transaction+identity ingestion path for users who accept the Kaggle competition rules.

## 1. Data and split

- Transactions: **590,540**
- Modeling features: **69**
- Fraud prevalence: **3.50%**
- Chronological split: **413,378 train / 88,581 validation / 88,581 test**

The split preserves time ordering so the final test set behaves more like future traffic than a random split would.

## 2. Model comparison

| Model | Split | ROC-AUC | PR-AUC | Brier |
|---|---|---:|---:|---:|
| Logistic baseline | validation | 0.7633 | 0.2247 | 0.2405 |
| LightGBM raw | validation | 0.9464 | 0.6312 | 0.0420 |
| LightGBM + Platt calibration | validation | 0.9464 | 0.6312 | 0.0192 |
| LightGBM + Platt calibration | test | **0.9332** | **0.5728** | **0.0212** |

### Interpretation

- LightGBM materially improves ranking quality over the interpretable logistic baseline.
- Calibration does not change ROC-AUC or PR-AUC because those metrics depend on ranking, but it sharply improves the Brier score, which matters when probabilities are used for cost-sensitive decisions.
- The test metrics are lower than validation metrics, which is expected under a future-like chronological holdout and is more credible than reporting only the best validation score.

## 3. Cost-sensitive decision policy

The illustrative policy assumes:

- missed fraud cost = **$200**
- false-decline cost = **$8**

The validation set selected a decline threshold of **0.03385**.

### Test performance at selected threshold

- Fraud recall: **83.43%**
- Precision: **19.35%**
- False-positive rate: **12.54%**
- Expected scenario cost: **$187,968**

At a naive 0.50 threshold:

- Fraud recall: **39.51%**
- Precision: **75.89%**
- False-positive rate: **0.45%**
- Expected scenario cost: **$376,096**

Under the stated cost assumptions, the selected operating point reduces expected scenario cost by **50.02%** versus the arbitrary 0.50 threshold.

This is a scenario result, not a claim of real company dollar savings. If business costs change, the optimal threshold changes.

## 4. Monitoring

Validation-to-test score PSI: **0.0023**.

This indicates very limited score-distribution drift across these two chronological windows. In production, PSI would be monitored together with feature drift, approval/review/decline rates, delayed-label recall/precision, and calibration.

## 5. Feature signals

Highest LightGBM feature importances include:

1. `card1_historical_fraud_rate`
2. `transaction_dt`
3. `card1_txn_count`
4. `card1_avg_amt`
5. `card1`
6. `transaction_amt`
7. `addr1`
8. `amt_vs_card_avg_ratio`
9. `card2`
10. `C13`

These signals reinforce an important fraud-modeling idea: behavior relative to an entity's historical baseline can be more informative than the raw transaction amount alone.

## 6. Main takeaways

1. **Ranking quality and decision quality are different.** ROC/PR-AUC evaluate ordering; calibration and thresholding determine operational actions.
2. **PR-AUC is the better headline metric for rare fraud.** Accuracy would be dominated by legitimate transactions.
3. **A 0.50 threshold is arbitrary.** The operating point should come from business costs and review capacity.
4. **Calibration matters.** If probabilities drive decisions, a well-ranked but poorly calibrated model can produce bad policies.
5. **Temporal validation is more realistic.** Fraud patterns evolve, so future-like holdouts are more informative than random splits.
6. **Production monitoring is part of the model.** Feature/score drift and delayed-label performance are required after launch.

## Interview narrative

> I framed fraud as an asymmetric decision problem rather than a generic classifier. I preserved time ordering, compared an interpretable logistic baseline with LightGBM, calibrated probabilities with Platt scaling, selected an operating threshold from explicit fraud-loss and false-decline costs, evaluated that threshold on a future-like holdout, and added drift metrics to connect offline modeling to production monitoring.
