# Full Fraud Analysis Results

These results were computed on the full **590,540-row** public model-ready IEEE-CIS feature table derived from the original Kaggle/Vesta data. The repository also preserves a raw transaction+identity ingestion path for users who accept the Kaggle competition rules.

## 1. Data and split

- Transactions: **590,540**
- Modeling features: **69**
- Fraud prevalence: **3.50%**
- Chronological split: **413,378 train / 88,581 validation / 88,581 test**

The split preserves time ordering so the final test set behaves more like future traffic than a random split would.

### Important implementation boundary

The verified full-data run used a **model-ready feature table** that already contains richer historical/entity signals. The current lightweight raw-ingestion feature module in `src/fraud_ds/features.py` does not yet reconstruct every one of those richer historical features from the original raw Kaggle files.

Therefore:

- the metrics below are verified outputs from the model-ready-table path;
- the raw path demonstrates ingestion, temporal validation, core feature engineering, training, calibration, policy, serving, and monitoring structure;
- the two paths are not yet one identical end-to-end feature-generation pipeline.

A future improvement is to reconstruct the richer historical/entity features from raw data using strictly past-only aggregation so that future labels cannot leak into earlier transactions.

## 2. Model comparison

| Model | Split | ROC-AUC | Average precision | Brier |
|---|---|---:|---:|---:|
| Logistic baseline | validation | 0.7633 | 0.2247 | 0.2405 |
| LightGBM raw | validation | 0.9464 | 0.6312 | 0.0420 |
| LightGBM + Platt calibration | validation | 0.9464 | 0.6312 | 0.0192 |
| LightGBM + Platt calibration | test | **0.9332** | **0.5728** | **0.0212** |

`average_precision_score` is the implementation used for the precision-recall summary metric. It is commonly discussed informally as PR-AUC, but it is more precise to call the reported value **average precision (AP)** rather than trapezoidal PR-AUC.

### Interpretation

- LightGBM materially improves ranking quality over the interpretable logistic baseline.
- Calibration does not change ROC-AUC or AP because those metrics depend primarily on ranking, but it sharply improves the Brier score, which matters when probabilities are used for cost-sensitive decisions.
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

This is a scenario result, not a claim of real company dollar savings. If business costs, review capacity, or calibration change, the preferred operating threshold can change as well.

## 4. Monitoring

Validation-to-test score PSI: **0.0023**.

This indicates very limited score-distribution drift across these two chronological windows. PSI is a **distribution-shift signal**, not proof that model performance is unchanged.

A real production monitoring setup would normally include several layers:

- **Data quality:** schema, missingness, ranges, duplicates, category frequencies
- **Feature / score drift:** PSI or related distribution-distance measures
- **Delayed-label performance:** AP, ROC-AUC, recall, precision, Brier score, calibration
- **Business outcomes:** approval rate, review volume, false-decline cost, fraud loss / chargebacks
- **Slice monitoring:** important segments such as country, device, merchant, or risk bands

Fraud labels may arrive days or weeks after the original transaction because chargebacks or investigations take time. Therefore, feature and score monitoring can run immediately, while label-dependent performance monitoring must wait until labels are sufficiently mature.

## 5. Feature signals

Highest saved LightGBM feature importances include:

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

These are model feature importances, not causal effects. Correlated variables can also share or mask importance. Permutation importance and SHAP would be reasonable diagnostic extensions, but they are **not currently implemented** in this repository.

## 6. Main takeaways

1. **Ranking quality and decision quality are different.** ROC-AUC / average precision evaluate ordering; calibration and thresholding determine operational actions.
2. **Average precision is the better headline metric for rare fraud.** Accuracy would be dominated by legitimate transactions.
3. **A 0.50 threshold is arbitrary.** The operating point should come from business costs, review capacity, and calibrated probabilities.
4. **Calibration matters.** If probabilities drive decisions, a well-ranked but poorly calibrated model can produce bad policies.
5. **Temporal validation is more realistic.** Fraud patterns evolve, so future-like holdouts are more informative than random splits.
6. **Monitoring is multi-layered.** Distribution drift can be observed before labels return; true performance requires mature labels.
7. **The verified model-ready-table path and current raw feature-generation path are not yet fully unified.** That boundary is explicit rather than hidden.

## Interview narrative

> I framed fraud as an asymmetric decision problem rather than a generic classifier. On a 590,540-row model-ready IEEE-CIS-derived table, I preserved time ordering, compared an interpretable logistic baseline with LightGBM, calibrated probabilities with Platt scaling, selected an operating threshold from explicit fraud-loss and false-decline costs, evaluated that threshold on a future-like holdout, and added drift-oriented monitoring components. The verified full-data run used richer pre-engineered historical/entity features, while the current raw-ingestion module reconstructs the core feature pipeline but not yet every historical feature from scratch.
