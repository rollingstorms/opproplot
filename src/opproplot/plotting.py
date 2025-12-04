from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt

from .core import OperatingProfile, compute_operating_profile


def operating_profile_plot(
    y_true,
    y_score,
    bins: int = 40,
    score_range=None,
    show_accuracy: bool = True,
    ax: Optional[plt.Axes] = None,
):
    """
    Plot an operating profile: stacked score histogram + TPR/FPR(/accuracy).

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth binary labels.
    y_score : array-like of shape (n_samples,)
        Predicted scores or probabilities.
    bins : int, default=40
        Number of bins for the score histogram.
    score_range : tuple(float, float) or None, default=None
        Range for the score histogram. If None, inferred from scores.
    show_accuracy : bool, default=True
        Whether to plot accuracy as an additional dashed curve.
    ax : matplotlib.axes.Axes or None, default=None
        Axis to plot on. If None, a new figure and axis are created.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax_hist : matplotlib.axes.Axes
        Axis with the histogram (left y-axis).
    ax_metric : matplotlib.axes.Axes
        Axis with the metric curves (right y-axis).
    """
    profile = compute_operating_profile(
        y_true=y_true, y_score=y_score, bins=bins, score_range=score_range
    )

    if ax is None:
        fig, ax_hist = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.get_figure()
        ax_hist = ax

    widths = profile.edges[1:] - profile.edges[:-1]
    ax_hist.bar(
        profile.mids,
        profile.neg_hist,
        width=widths,
        align="center",
        label="Negatives",
    )
    ax_hist.bar(
        profile.mids,
        profile.pos_hist,
        width=widths,
        align="center",
        bottom=profile.neg_hist,
        label="Positives",
        alpha=0.7,
    )

    ax_hist.set_xlabel("Score bins (threshold midpoints)")
    ax_hist.set_ylabel("Count per bin")

    ax_metric = ax_hist.twinx()
    ax_metric.plot(profile.mids, profile.tpr, label="TPR (Recall)")
    ax_metric.plot(profile.mids, profile.fpr, label="FPR")

    if show_accuracy:
        ax_metric.plot(profile.mids, profile.accuracy, label="Accuracy", linestyle="--")

    ax_metric.set_ylabel("Metric value")

    ax_hist.legend(loc="upper left")
    ax_metric.legend(loc="lower right")
    ax_hist.set_title("Opproplot: Operating Profile")

    fig.tight_layout()
    return fig, ax_hist, ax_metric
