# Interview Guide — Production Fraud Risk Decision System

## 60-second walkthrough
I framed fraud detection as a cost-sensitive operating decision, not an accuracy problem. On a 590,540-row IEEE-CIS-derived table, I used chronological train/validation/test windows to avoid leaking future patterns. I compared an interpretable logistic baseline with LightGBM, calibrated probabilities, selected a threshold using explicit missed-fraud and false-decline costs, and expressed the outcome as approve/review/decline operations. I also included API, Docker, tests, CI, and score-drift monitoring.

## Know these ideas
- **Why temporal splits:** fraud behavior changes over time; random splitting can let the model learn future patterns.
- **Why PR-AUC:** with only 3.5% fraud, accuracy and ROC-AUC alone obscure minority-class retrieval.
- **Calibration:** thresholds depend on probabilities; a well-ranked but poorly calibrated score can make poor decisions.
- **Threshold economics:** lower thresholds catch more fraud but create more customer friction.
- **PSI:** compares score distributions across periods; it signals drift, not automatically model failure.

## Likely questions
**Why not classify at 0.50?**  
0.50 is arbitrary. The operating threshold should reflect fraud loss, false-decline cost, review capacity, and calibrated probability quality.

**Why use LightGBM after logistic regression?**  
Logistic regression is an interpretable benchmark. Gradient boosting can capture nonlinear interactions in high-dimensional tabular data; I retained the baseline to quantify whether complexity earned its place.

## Reproduce and learn
1. Read `RESULTS.md` and identify the validation-selected threshold.
2. Explain the expected-cost equation using a 2×2 confusion matrix.
3. Run the test suite and trace the temporal split code.
4. Use the API once with a sample request.
5. Explain one false-positive and one false-negative operational consequence.

## Honest boundary
The dollar-cost scenario is illustrative. It is not a real-company savings claim.