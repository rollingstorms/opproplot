# API Reference

## compute_operating_profile

```python
from opproplot import compute_operating_profile
profile = compute_operating_profile(y_true, y_score, bins=40, score_range=(0, 1))
```

- `y_true`: array-like of shape (n_samples,), binary labels.
- `y_score`: array-like of shape (n_samples,), predicted scores or probabilities.
- `bins`: number of score bins (default 40).
- `score_range`: tuple or None. If None, uses min/max of scores.

Returns an `OperatingProfile` dataclass with:
- `edges`, `mids`, `pos_hist`, `neg_hist`, `tpr`, `fpr`, `accuracy`.

## operating_profile_plot

```python
from opproplot import operating_profile_plot
fig, ax_hist, ax_metric = operating_profile_plot(y_true, y_score, bins=30, show_accuracy=True)
```

- `show_accuracy`: include the dashed accuracy curve (default True).
- `ax`: optional Matplotlib axis to draw on; otherwise creates a new figure.

Returns `(fig, ax_hist, ax_metric)` for further styling or saving.
