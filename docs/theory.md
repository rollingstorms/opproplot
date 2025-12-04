# Theory: The Geometry of Thresholds

Opproplot treats thresholding as a geometric object over score space. For a scoring function f(x) and threshold t, the decision rule is

h_t(x) = 1{f(x) >= t}.

## Distributions

- p(s | Y=1) and p(s | Y=0) are estimated with class-conditional histograms.
- Midpoints of bins act as candidate thresholds.

## Metrics as cumulative integrals

- True Positive Rate: TPR(t) = P(f(X) >= t | Y=1).
- False Positive Rate: FPR(t) = P(f(X) >= t | Y=0).
- Accuracy: Acc(t) = [TP(t) + TN(t)] / (P + N).

These are computed in a single pass over scores by sorting once and evaluating cumulative counts at the bin midpoints.

## Why this view

- Links the score distribution to threshold outcomes directly.
- Shows the full family of operating points without switching plots.
- Works for imbalanced data: histogram heights reveal prevalence while TPR/FPR curves show trade-offs.
