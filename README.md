# Production Fraud Risk Decision System

End-to-end Data Science / ML Engineering portfolio project built on the **IEEE-CIS Fraud Detection (Vesta)** e-commerce payment dataset.

The goal is not to maximize classification accuracy. Fraud decisions create asymmetric business costs:

- **False negative:** fraud is approved -> fraud loss / chargeback.
- **False positive:** a legitimate payment is blocked -> lost revenue, customer friction, support cost.

The project therefore treats fraud prediction as a **risk decision system**:

`transaction data -> temporal validation -> feature engineering -> gradient boosting -> probability calibration -> cost-sensitive threshold -> approve/review/decline policy -> serving -> monitoring`

## Verified full-data highlights

The verified modeling analysis was executed on a **590,540-row public model-ready IEEE-CIS feature table derived from the original Kaggle/Vesta data**. See [`RESULTS.md`](RESULTS.md) and `results/*.json` for persisted outputs.

- Fraud prevalence: **3.50%**
- Chronological split: **413,378 train / 88,581 validation / 88,581 test**
- Logistic baseline validation average precision: **0.2247**
- LightGBM validation average precision: **0.6312**
- Calibrated LightGBM test ROC-AUC: **0.9332**
- Calibrated LightGBM test average precision: **0.5728**
- Test Brier score after Platt calibration: **0.0212**
- Validation-selected decline threshold: **0.03385**
- Test fraud recall at that threshold: **83.43%**
- Test precision: **19.35%**
- Test false-positive rate: **12.54%**
- Under the explicit illustrative cost assumptions of **$200 per missed fraud** and **$8 per false decline**, expected test cost was **50.02% lower** than using an arbitrary 0.50 threshold
- Validation-to-test score PSI: **0.0023**, indicating little score-distribution drift across the two holdout windows

`average_precision_score` is used as the precision-recall summary metric in the code/results; this is often informally described as PR-AUC, but average precision and trapezoidal PR-AUC are not exactly the same numerical definition.

The cost reduction is a scenario result under stated assumptions, **not a claim of real-company dollar savings**.

## Dataset and two execution paths

Kaggle competition: `ieee-fraud-detection`

Key raw files:
- `train_transaction.csv`
- `train_identity.csv`

The transaction and identity tables join on `TransactionID`. The target is `isFraud`. Raw competition data is not committed because download requires accepting the Kaggle competition rules.

The repository contains **two related but not yet fully unified paths**:

### Path A — raw-ingestion pipeline implemented in this repository

After Kaggle authorization:

```bash
python scripts/download_data.py
```

The current raw pipeline loads transaction/identity files, checks schema and transaction uniqueness, preserves chronology, and creates the core engineered features implemented in `src/fraud_ds/features.py`, including:

- `log_amount`
- `day_index`
- `hour`
- `is_night`

It also uses selected transaction/card/address/count/duration/category fields that already exist in the input data.

### Path B — verified full modeling analysis

The persisted 590,540-row results were produced from a **public model-ready IEEE-CIS-derived feature table**. That table already contains a richer feature set, including entity-history signals such as:

- `card1_historical_fraud_rate`
- `card1_txn_count`
- `card1_avg_amt`
- `amt_vs_card_avg_ratio`
- `email_historical_fraud_rate`

These richer signals were used by the verified full-data model and appear in the saved feature-importance output.

### Implementation boundary

The current repository does **not yet reconstruct every richer historical/entity feature in Path B from the raw Kaggle files**. In other words, the verified model-ready table and the raw feature-generation module are related, but they are not currently one identical end-to-end feature pipeline.

A natural next engineering step is to move those historical/entity features into the raw pipeline using strictly **past-only aggregations** (for example, `shift(1)` / expanding history) so that the same feature definitions can be reproduced from raw data without target or future leakage.

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

Historical/entity features should also be computed using only information available **before** each transaction. A full-population groupby using future labels would create target leakage.

### 2. Baseline vs production-style model
- **Logistic Regression** provides an interpretable benchmark.
- **LightGBM** captures nonlinear interactions common in high-dimensional tabular fraud data.

On validation data, average precision improved from **0.2247** for the logistic baseline to **0.6312** for LightGBM.

### 3. Probability calibration
Platt calibration reduced validation Brier score from **0.0420** to **0.0192** while leaving ranking metrics unchanged. This matters because the model uses class weighting and because policy thresholds are applied to predicted probabilities, not just arbitrary ranking scores.

Calibration asks a different question from ranking: among transactions assigned approximately 10% fraud probability, does the observed fraud rate also approach 10%?

### 4. Rare-event metrics
- ROC-AUC for overall ranking context
- **Average precision / precision-recall performance** as the headline rare-class ranking metric
- Brier score / log loss for probability quality
- fraud recall, precision, false-positive rate, and expected decision cost at the operating threshold

Accuracy is intentionally not the headline metric because **96.5%** of observations are legitimate transactions.

### 5. Cost-sensitive decision policy
The threshold is selected by minimizing:

```text
expected_cost = false_negatives * fraud_miss_cost
              + false_positives * false_decline_cost
```

At the selected threshold, the model catches substantially more fraud than a 0.50 threshold, at the cost of more false positives. That trade-off is deliberate: the operating point depends on economics, review capacity, and calibrated probability quality rather than a universal probability cutoff.

A three-way policy separates scoring from operations:

- approve
- manual review
- decline

## Feature signals and interpretation

Highest saved LightGBM importances from the verified full-data analysis include historical card fraud rate, transaction time, card transaction frequency, average card amount, transaction amount, address information, and amount relative to a card's historical average.

The takeaway is that **behavior relative to an entity's history can be more informative than raw transaction amount alone**.

The saved importance values are model feature importances; they do **not** imply causality. Correlated features can also split importance between one another.

Permutation importance and SHAP are useful possible extensions for stronger diagnostics and local explanations, but **they are not currently implemented in this repository**.

## Serving and monitoring path

```text
raw CSVs OR model-ready feature table
   |
   v
schema / data-quality checks
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
model + metadata persistence / MLflow tracking
   |
   v
FastAPI /predict
   |
   v
Docker + CI
   |
   v
feature / score drift checks
```

The repository includes a PSI utility for comparing reference and current numeric distributions. PSI is a **drift signal**, not proof that the model has failed.

A real production monitoring system would normally go beyond PSI and monitor multiple layers:

- **Data quality:** schema, missingness, ranges, duplicates, category frequency
- **Feature / score drift:** PSI or other distribution-distance measures
- **Delayed-label model performance:** average precision, ROC-AUC, recall, precision, Brier score, calibration
- **Business outcomes:** approval rate, review volume, false-decline cost, fraud loss / chargebacks
- **Slices:** performance and drift by important segments such as country, device, merchant, or risk band

Fraud labels are often delayed because chargebacks or investigations can arrive days or weeks after the transaction. For that reason, distribution/data-quality monitoring can run immediately, while label-dependent performance monitoring must wait until labels are sufficiently mature.

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

1. **Ranking and decisions are different problems.** Average precision / ROC-AUC measure ranking quality; calibration and thresholds determine actions.
2. **A 0.50 threshold is arbitrary.** The economically appropriate threshold can be much lower for high-cost missed fraud.
3. **Calibration matters when scores become probabilities used in policy.**
4. **Temporal holdouts are more credible for fraud than random splits.**
5. **Monitoring is multi-layered.** Distribution drift can be checked before labels arrive; true model performance requires mature labels.
6. **The raw feature pipeline and verified model-ready-table path are not yet fully unified.** That boundary is documented explicitly rather than hidden.

## Resume-ready description

> Built an end-to-end fraud risk decision system across 590K+ IEEE-CIS e-commerce transactions, improving validation average precision from 0.225 with logistic regression to 0.631 with LightGBM and achieving 0.573 average precision on a chronological test set; calibrated probabilities and optimized a cost-sensitive operating threshold that reduced expected scenario cost by 50% versus a naive 0.50 cutoff under explicit cost assumptions.

## Current scope vs future work

### Implemented / verified
- chronological holdout design
- logistic baseline and LightGBM comparison in the verified analysis
- class-imbalance-aware evaluation
- Platt probability calibration
- cost-sensitive threshold selection
- approve/review/decline policy logic
- model/metadata persistence and MLflow logging path
- FastAPI scoring endpoint
- Docker / tests / CI
- PSI-based drift utility
- persisted full-data metrics and feature-importance outputs

### Not yet fully implemented
- end-to-end reconstruction of all historical/entity features from raw Kaggle files
- permutation importance
- SHAP explanations
- a continuously running production monitoring service
- automated retraining / recalibration triggers
- live bank/payment-company deployment or real traffic

## Why this is not a toy project

The project emphasizes temporal leakage, rare-event evaluation, probability calibration, economics of false decisions, operational review capacity, modular pipelines, API contracts, containerization, automated tests, persisted full-data outputs, and drift-aware monitoring design—not just a leaderboard score.

## Learn it for interviews

Use the project-specific [Interview Guide](docs/INTERVIEW_GUIDE.md) for a 60-second walkthrough, key concepts, likely questions, reproducible study steps, and the honest boundary of the work.
