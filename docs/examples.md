# Examples

Use these patterns to compare models and datasets.

## Breast cancer (scikit-learn)

- Load `sklearn.datasets.load_breast_cancer`.
- Train a logistic regression or gradient boosting model.
- Plot the operating profile on the test split to inspect separability.

## Fraud-like imbalance

- Simulate or load an imbalanced dataset.
- Compare a calibrated model vs an overconfident one.
- Observe how class imbalance alters histogram heights and accuracy peaks.

## Good vs bad model

- Train two models on the same data.
- Plot both operating profiles side by side.
- Look for:
  - Separation of score distributions.
  - Lower FPR for the same TPR.
  - Stability of accuracy across thresholds.

Swap in your own datasets; the plotting API stays the same.
