# Step-by-Step Learning Guide — Production Fraud Risk Decision System

This guide explains the project in the order you should learn and present it.

## 1. Business framing

The goal is not simply to classify fraud. A fraud system makes costly decisions:

- false negative: fraud is approved
- false positive: legitimate payment is blocked

So the real objective is:

> maximize fraud detection while controlling customer friction and expected decision cost.

## 2. Data

The full modeling analysis uses **590,540 IEEE-CIS transactions** with **69 modeling features**.

Fraud prevalence: **3.50%**.

Because only about 3.5% of rows are fraud, accuracy is not a useful headline metric: a model predicting every transaction as legitimate would already be about 96.5% accurate.

## 3. Why chronological splitting?

Fraud patterns evolve over time. A random split lets future attack patterns leak into training and can make performance look unrealistically strong.

The project preserves chronology:

- Train: **413,378**
- Validation: **88,581**
- Test: **88,581**

The test set acts as a future-like holdout.

## 4. Feature engineering

The model uses both raw and behavior-relative signals, including:

- transaction amount
- card/address/device information
- hour of day
- historical card transaction count
- historical average amount
- historical fraud rate
- transaction amount relative to card history

High-importance features include:

1. card historical fraud rate
2. transaction time
3. card transaction count
4. card average amount
5. raw card identifier
6. transaction amount
7. address
8. amount relative to card average

### Takeaway

Fraud often depends on whether a transaction is unusual **relative to historical behavior**, not just whether the raw amount is large.

## 5. Interpretable baseline

A Logistic Regression baseline was trained first.

Validation metrics:

- ROC-AUC: **0.7633**
- PR-AUC: **0.2247**
- Brier score: **0.2405**

### Why start simple?

A baseline tells us whether added model complexity actually creates value. Without a baseline, a sophisticated model score has no context.

## 6. LightGBM

A gradient-boosted tree model was then trained for nonlinear tabular interactions.

Validation metrics:

- ROC-AUC: **0.9464**
- PR-AUC: **0.6312**
- Brier: **0.0420**

The large PR-AUC improvement shows that LightGBM ranks rare fraud cases much better than the linear baseline.

## 7. Why PR-AUC?

ROC-AUC is useful, but under strong imbalance it can look good even when positive-class precision is weak.

PR-AUC focuses on:
- precision among flagged transactions
- recall of actual fraud

That makes it more informative for a 3.5%-prevalence fraud problem.

## 8. Probability calibration

The raw LightGBM probabilities were calibrated using Platt scaling.

Validation Brier score improved:

- before calibration: **0.0420**
- after calibration: **0.0192**

PR-AUC stayed **0.6312** because calibration changes probability scale, not ranking.

### Why calibration matters

If probabilities feed a cost-sensitive threshold, a score of 0.10 should mean something closer to a 10% risk level. Good ranking alone is not enough.

## 9. Final chronological test performance

Calibrated LightGBM test metrics:

- ROC-AUC: **0.9332**
- PR-AUC: **0.5728**
- Brier score: **0.0212**

The test score is lower than validation, which is normal and more credible than reporting only the best validation result.

## 10. Why not threshold at 0.50?

There is nothing mathematically special about 0.50.

The project uses an illustrative business-cost model:

- missed fraud: **$200**
- false decline: **$8**

For each candidate threshold:

`expected cost = FN * 200 + FP * 8`

The threshold minimizing validation cost was **0.03385**.

## 11. Test behavior at selected threshold

At threshold **0.03385**:

- Fraud recall: **83.43%**
- Precision: **19.35%**
- False-positive rate: **12.54%**
- Scenario expected cost: **$187,968**

At threshold **0.50**:

- Fraud recall: **39.51%**
- Precision: **75.89%**
- False-positive rate: **0.45%**
- Scenario expected cost: **$376,096**

Under the stated assumptions, the selected threshold reduces expected cost by **50.02%**.

### Critical interpretation

The lower threshold catches far more fraud but creates more legitimate false positives. Whether that trade-off is acceptable depends on real company economics and review capacity.

The 50% number is a **scenario result**, not real company savings.

## 12. Approve / review / decline policy

Instead of treating every high score identically, the production design supports three actions:

- approve: low risk
- review: uncertain/high enough risk for manual investigation
- decline: highest risk

This separates model scoring from operational decision-making.

## 13. Monitoring

Validation-to-test score PSI: **0.0023**.

This is low, suggesting little score-distribution drift between these windows.

A real production system would monitor:

- feature distributions
- missing values
- fraud-score distribution
- approve/review/decline rates
- delayed-label fraud recall
- calibration
- segment-level false positives

## 14. API / engineering layer

The repository includes:

- FastAPI inference service
- Docker container
- MLflow experiment/artifact path
- automated unit tests
- GitHub Actions CI
- monitoring utilities

The point is to demonstrate that the model is part of a system, not just a notebook.

## 15. Main project takeaways

1. Fraud detection is a decision problem, not only a classification problem.
2. Class imbalance changes which metrics matter.
3. Temporal validation makes offline evaluation more realistic.
4. LightGBM materially outperformed the logistic baseline.
5. Calibration improved probability quality without changing ranking.
6. Threshold selection should follow business costs, not default 0.50.
7. Better recall comes with customer-friction trade-offs.
8. Production monitoring is necessary because fraud patterns change.

## 16. 60-second interview version

> I built an end-to-end fraud risk decision system on 590,540 IEEE-CIS transactions with a 3.5% fraud rate. Because fraud is a rare and time-evolving event, I used a chronological train-validation-test split and focused on PR-AUC rather than accuracy. I started with logistic regression as an interpretable baseline, which achieved 0.225 validation PR-AUC, then LightGBM improved it to 0.631. I calibrated the LightGBM probabilities with Platt scaling, which reduced Brier score from 0.042 to 0.019. Instead of using a default 0.50 threshold, I optimized the threshold under explicit missed-fraud and false-decline costs. On the future-like test set, the selected threshold captured 83.4% of fraud and reduced expected scenario cost by about 50% versus the 0.50 cutoff under those assumptions. I also added an approve-review-decline policy, FastAPI, Docker, CI, and drift monitoring to connect offline modeling to production decisions.

## 17. Questions you should be ready for

- Why not use accuracy?
- Why chronological instead of random split?
- Why Logistic Regression first?
- Why LightGBM for tabular fraud?
- ROC-AUC vs PR-AUC?
- Why does calibration matter?
- Why did calibration improve Brier but not PR-AUC?
- Why is 0.50 arbitrary?
- How would you estimate real false-decline cost?
- What if manual review capacity is limited?
- What is PSI?
- What would you monitor after deployment?
- Which features risk leakage and how would you prevent it?
