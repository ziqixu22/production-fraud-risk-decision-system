# Production Fraud Risk Decision System

End-to-end Data Science / ML Engineering portfolio project built on the **IEEE-CIS Fraud Detection (Vesta)** e-commerce payment dataset.

The goal is not to maximize classification accuracy. Fraud decisions create asymmetric business costs:

- **False negative:** fraud is approved -> fraud loss / chargeback.
- **False positive:** a legitimate payment is blocked -> lost revenue, customer friction, support cost.

The project therefore treats fraud prediction as a **risk decision system**:

`transaction + identity data -> leakage-safe temporal validation -> feature engineering -> gradient boosting -> probability calibration -> cost-sensitive threshold -> approve/review/decline policy -> FastAPI -> Docker -> monitoring`

## Verified full-data highlights

The full modeling analysis was executed on a **590,540-row public model-ready IEEE-CIS feature table derived from the original Kaggle/Vesta data**. See [`RESULTS.md`](RESULTS.md) and `results/*.json` for persisted outputs.

- Fraud prevalence: **3.50%**
- Chronological split: **413,378 train / 88,581 validation / 88,581 test**
- Logistic baseline validation PR-AUC: **0.2247**
- LightGBM validation PR-AUC: **0.6312**
- Calibrated LightGBM test ROC-AUC: **0.9332**
- Calibrated LightGBM test PR-AUC: **0.5728**
- Test Brier score after Platt calibration: **0.0212**
- Validation-selected decline threshold: **0.03385**
- Test fraud recall at that threshold: **83.43%**
- Test precision: **19.35%**
- Test false-positive rate: **12.54%**
- Under the explicit illustrative cost assumptions of **$200 per missed fraud** and **$8 per false decline**, expected test cost was **50.02% lower** than using an arbitrary 0.50 threshold
- Validation-to-test score PSI: **0.0023**, indicating little score-distribution drift across the two holdout windows

The cost reduction is a scenario result under stated assumptions, **not a claim of real-company dollar savings**.

## Dataset

Kaggle competition: `ieee-fraud-detection`

Key raw files:
- `train_transaction.csv`
- `train_identity.csv`

The transaction and identity tables join on `TransactionID`. The target is `isFraud`. Raw competition data is not committed because download requires accepting the Kaggle competition rules.

The repository supports the raw ingestion path after authorization:

```bash
python scripts/download_data.py
```

For the verified full-data results above, the analysis used a public model-ready feature table derived from the original IEEE-CIS/Vesta competition data so the complete 590,540-row modeling stage could be executed reproducibly without embedding restricted raw files in the repository.

## Business questions

1. Can the model rank fraudulent transactions well under severe class imbalance?
2. Are predicted probabilities calibrated enough to support risk decisions?
3. Which threshold minimizes expected fraud-loss + false-decline cost?
4. What happens to fraud recall when customer friction is constrained?
5. Which signals contribute most to model decisions?
6. How would feature and score drift be detected after deployment?

## Modeling design

### 1. Leakage-safe temporal validation
`TransactionDT` is used to preserve chronology. Random splitting can leak future fraud patterns backward and make offline performance look more optimistic than real deployment.

### 2. Baseline vs production-style model
- **Logistic Regression** provides an interpretable benchmark.
- **LightGBM** captures nonlinear interactions common in high-dimensional tabular fraud data.

On validation data, PR-AUC improved from **0.2247** for the logistic baseline to **0.6312** for LightGBM.

### 3. Probability calibration
Platt calibration reduced validation Brier score from **0.0420** to **0.0192** while leaving ranking metrics unchanged. This matters because thresholds are chosen from predicted probabilities.

### 4. Rare-event metrics
- ROC-AUC for ranking context
- **PR-AUC** as the headline rare-class ranking metric
- Brier score / log loss for probability quality
- fraud recall, precision, false-positive rate, and expected decision cost at the operating threshold

Accuracy is intentionally not the headline metric because **96.5%** of observations are legitimate transactions.

### 5. Cost-sensitive decision policy
The threshold is selected by minimizing:

```text
expected_cost = false_negatives * fraud_miss_cost
              + false_positives * false_decline_cost
```

At the selected threshold, the model catches substantially more fraud than a 0.50 threshold, at the cost of more false positives. That trade-off is deliberate: the operating point depends on economics, not a universal probability cutoff.

A three-way policy separates scoring from operations:

- approve
- manual review
- decline

## Feature signals

High-importance signals include historical card fraud rate, transaction time, card transaction frequency, average card amount, transaction amount, address information, and amount relative to a card's historical average.

The takeaway is that **behavior relative to an entity's history can be more informative than raw transaction amount alone**.

## Production path

```text
raw CSVs / model-ready feature table
   |
   v
schema validation
   |
   v
chronological split
   |
   v
feature pipeline
   |
   v
Logistic baseline + LightGBM
   |
   v
probability calibration
   |
   v
cost-sensitive threshold
   |
   v
MLflow model + metadata
   |
   v
FastAPI /predict
   |
   v
Docker + CI
   |
   v
feature / score drift monitoring
```

## Repository structure

```text
.
├── README.md
├── RESULTS.md
├── results/
│   ├── fraud_metrics.json
│   └── feature_importance_top25.csv
├── configs/model.yaml
├── scripts/download_data.py
├── src/fraud_ds/
│   ├── data.py
│   ├── features.py
│   ├── metrics.py
│   ├── policy.py
│   ├── monitoring.py
│   ├── train.py
│   └── api.py
├── tests/
├── docs/
├── Dockerfile
├── .github/workflows/ci.yml
└── pyproject.toml
```

## Key takeaways

1. **Ranking and decisions are different problems.** PR-AUC measures ranking quality; calibration and thresholds determine actions.
2. **A 0.50 threshold is arbitrary.** The economically appropriate threshold can be much lower for high-cost missed fraud.
3. **Calibration matters when scores become probabilities used in policy.**
4. **Temporal holdouts are more credible for fraud than random splits.**
5. **Model monitoring is part of the system, not an afterthought.**

## Resume-ready description

> Built an end-to-end fraud risk decision system across 590K+ IEEE-CIS e-commerce transactions, improving validation PR-AUC from 0.225 with logistic regression to 0.631 with LightGBM and achieving 0.573 PR-AUC on a chronological test set; calibrated probabilities and optimized a cost-sensitive operating threshold that reduced expected scenario cost by 50% versus a naive 0.50 cutoff under explicit cost assumptions.

## Why this is not a toy project

The project emphasizes temporal leakage, rare-event evaluation, probability calibration, economics of false decisions, operational review capacity, modular pipelines, API contracts, containerization, automated tests, persisted full-data outputs, and drift monitoring—not just a leaderboard score.


## Learn it for interviews

Use the project-specific [Interview Guide](docs/INTERVIEW_GUIDE.md) for a 60-second walkthrough, key concepts, likely questions, reproducible study steps, and the honest boundary of the work.
