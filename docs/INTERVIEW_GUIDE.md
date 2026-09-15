# Interview Guide — Production Fraud Risk Decision System

## 60-second walkthrough
I framed fraud detection as a cost-sensitive operating decision, not an accuracy problem. On a 590,540-row model-ready IEEE-CIS-derived table, I used chronological train/validation/test windows to mimic future traffic, compared an interpretable logistic baseline with LightGBM, calibrated probabilities, selected a threshold using explicit missed-fraud and false-decline costs, and expressed the outcome as approve/review/decline operations. I also included API, Docker, tests, CI, and PSI-based score-drift monitoring components.

## Know these ideas
- **Why temporal splits:** fraud behavior changes over time; random splitting can leak future patterns backward or create an unrealistically easy validation setting.
- **Why average precision / PR performance:** with only 3.5% fraud, accuracy and ROC-AUC alone can obscure minority-class retrieval quality.
- **Calibration:** ranking scores and true probabilities are different concepts; a well-ranked but poorly calibrated score can lead to poor cost-sensitive decisions.
- **Threshold economics:** lower thresholds catch more fraud but create more false positives / customer friction.
- **PSI:** compares score or feature distributions across periods; it signals distribution shift, not automatically model failure.
- **Delayed-label monitoring:** chargebacks or fraud investigations may arrive days or weeks later, so recall, precision, AP, Brier score, and calibration cannot always be evaluated immediately.

## Likely questions

**Why not classify at 0.50?**  
0.50 is an arbitrary convention. The operating threshold should reflect fraud loss, false-decline cost, review capacity, and calibrated probability quality.

**Why use LightGBM after logistic regression?**  
Logistic regression is an interpretable benchmark. Gradient boosting can capture nonlinear relationships and interactions in tabular fraud data; I retained the baseline to quantify whether additional complexity earned its place.

**Why do you need calibration after class weighting / boosted-tree scoring?**  
Class weighting helps the model pay attention to the rare fraud class and can improve ranking, but the resulting score scale is not guaranteed to equal the real-world fraud probability. Calibration maps model scores to probabilities that better match observed event rates, which matters when probabilities feed into cost-based policy decisions.

**What exactly was implemented from raw data?**  
The repository has a raw transaction/identity ingestion path with schema checks, chronology, core feature engineering, training, calibration, thresholding, and serving. The verified 590,540-row full-data results were produced from a public model-ready IEEE-CIS-derived feature table that already contained richer historical/entity features such as prior fraud rate, transaction count, and average card amount. The current raw feature module does not yet rebuild every one of those richer features from scratch.

**How would you prevent leakage in historical fraud features?**  
For transaction i, use only information available strictly before transaction i. For example, use time ordering plus shift/expanding or equivalent past-only aggregation. A full-population groupby mean using future fraud labels would leak target information backward.

**What would you improve next?**  
First, unify the raw pipeline and verified full-analysis feature set by reconstructing all historical/entity features with strictly past-only logic. Then add stronger diagnostics such as permutation importance or SHAP, and expand monitoring beyond PSI to include data quality, score drift, delayed-label performance, calibration, business outcomes, and segment-level monitoring.

## Reproduce and learn
1. Read `RESULTS.md` and explain the implementation boundary between the raw path and verified model-ready-table path.
2. Explain the expected-cost equation using a 2×2 confusion matrix.
3. Trace `data.py`, `features.py`, `metrics.py`, `policy.py`, and `monitoring.py`.
4. Explain why the reported precision-recall summary is average precision (`average_precision_score`).
5. Use the API once with a sample request and explain how calibrated probability becomes approve/review/decline.
6. Explain what can be monitored immediately versus what requires mature fraud labels.

## Honest boundary
- The dollar-cost scenario is illustrative, not a real-company savings claim.
- The verified full-data model used a richer model-ready feature table; the current raw pipeline does not yet reconstruct every historical/entity feature.
- Permutation importance and SHAP are not currently implemented.
- The repository demonstrates production-oriented components, but it is not a claim of live bank/payment-company deployment or live traffic.
