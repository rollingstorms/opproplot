# Opproplot

A compact operating profile plot for binary classifiers: stacked score histograms by class plus TPR/FPR/Accuracy curves at bin-midpoint thresholds. One view to understand every possible cutoff.

## Why Opproplot

- See score separation between classes directly.
- Trace how recall and false positives move as you slide the threshold.
- Spot the accuracy peak without losing visibility into the distribution.

## Install

```bash
pip install -e .
```

## Quickstart

```python
import numpy as np
from opproplot import operating_profile_plot

rng = np.random.default_rng(0)
y_true = rng.integers(0, 2, size=5000)
scores = rng.random(size=5000)

operating_profile_plot(y_true, scores, bins=30)
```

## Learn more

- [Getting started](getting_started.md): notebook-friendly walkthroughs.
- [Theory](theory.md): decision rules, distributions, and threshold geometry.
- [Examples](examples.md): real datasets and comparisons.
- [API](api.md): function reference and parameters.
- [Roadmap](roadmap.md): upcoming features.
