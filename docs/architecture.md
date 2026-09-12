# Architecture decisions

## Temporal validation
Fraud patterns, merchants, devices, and attack strategies evolve. Chronological validation is closer to deployment than a random split and reduces leakage from future behavior.

## Rare-event metrics
Accuracy is dominated by the legitimate-payment majority class. PR-AUC focuses on positive-class precision/recall; ROC-AUC remains useful for ranking context. Brier score and log loss evaluate probability quality.

## Calibration
Decision thresholds use predicted probabilities. A model can rank well yet produce poorly calibrated probabilities, so the selected model is calibrated on a held-out validation period.

## Cost-sensitive thresholding
A 0.5 threshold has no inherent business meaning. The decision threshold minimizes an explicit cost function combining missed-fraud and false-decline costs.

## Three-way decision policy
Approve/review/decline separates model risk scoring from operational capacity. In a real system, review thresholds should reflect queue capacity, expected transaction value, and risk appetite.

## API contract
Serving the model through an API forces an explicit feature schema and model-version response rather than leaving the result as a notebook-only artifact.

## Monitoring
After deployment, monitor input schema/missingness, feature drift, score-distribution drift, approval/review/decline rates, delayed-label performance, and calibration by segment.