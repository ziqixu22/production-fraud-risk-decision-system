# Interview guide

## Core questions you should be able to answer

1. Why is random splitting risky for fraud detection?
2. Why is accuracy misleading under class imbalance?
3. ROC-AUC vs PR-AUC: when does each matter?
4. Why does probability calibration matter for a risk decision system?
5. Why is the decision threshold a business decision rather than only an ML choice?
6. What are the economic consequences of false declines?
7. How would delayed fraud labels affect monitoring?
8. What would data leakage look like in this dataset?
9. How does a new transaction move through the API to a final decision?
10. Which types of drift should be monitored after deployment?

## Project story

Start with the business objective: minimize expected fraud loss while controlling legitimate-customer friction. Explain why you used chronological validation, why PR-AUC and calibration matter, how you converted probabilities into an approve/review/decline policy, and how the production components support reproducibility and monitoring.

## Trade-offs

### Recall vs customer friction
Lower thresholds catch more fraud but block more legitimate payments.

### Review capacity
A manual-review band allows uncertain transactions to be inspected without automatically declining all high-risk scores. Real thresholds should account for queue capacity.

### Complexity vs value
Gradient-boosted trees are a strong tabular baseline. A more complex model should be adopted only if incremental economic value justifies latency, maintenance, and explainability costs.

### Offline vs production
Offline PR-AUC is necessary but not sufficient. Production success also depends on stable feature definitions, latency, calibration, monitoring, and a feedback loop for delayed labels.

## What not to claim

Do not claim a real company's fraud savings, production SLA, AWS deployment, or live customer traffic unless those pieces are actually implemented and measured.