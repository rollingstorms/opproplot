# Examples

Use these patterns to compare models and datasets.

## Clear separation (breast cancer, scikit-learn)

- Load `sklearn.datasets.load_breast_cancer`.
- Train a logistic regression or gradient boosting model.
- Plot the operating profile on the test split to inspect separability.
- Interpretation: distributions are well separated; TPR stays high while FPR stays low across much of the threshold range.

## Ambiguous scores (overlapping normals)

- Simulate scores from two overlapping normal distributions with similar means/variance.
- Expect intertwined histograms and TPR/FPR curves that cross more frequently.
- Interpretation: thresholds are fragile; small shifts move a lot of examples between classes.

## Bumpy distributions (mixed pockets)

- Build a model that produces multi-modal scores (e.g., mixture components or segment-specific calibrations).
- Look for “bumps” in the histogram and corresponding inflections in TPR/FPR.
- Interpretation: localized score clusters may indicate subpopulations; thresholding there can create sharp metric changes.

Swap in your own datasets; the plotting API stays the same.
