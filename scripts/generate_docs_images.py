"""
Generate documentation images for Opproplot.

Creates:
- docs/assets/opproplot_hero.png
- docs/assets/opproplot_example.png
- docs/assets/opproplot_breast_cancer.png
"""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.datasets import load_breast_cancer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from opproplot import operating_profile_plot  # noqa: E402


ASSETS_DIR = Path("docs/assets")


def _ensure_assets_dir() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)


def generate_hero() -> None:
    rng = np.random.default_rng(2)
    y_true = rng.integers(0, 2, size=4000)
    scores = rng.normal(loc=y_true * 0.7 + 0.08, scale=0.3, size=4000)
    scores = np.clip(scores, 0, 1)

    fig, ax_hist, ax_metric = operating_profile_plot(
        y_true,
        scores,
        bins=24,
        show_accuracy=True,
        show_key=True,
        key_location="outside",
        show_grid=False,
        title="Operating Profile Plot",
    )

    # Minimal styling for hero
    for ax in (ax_hist, ax_metric):
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(labelbottom=False, labelleft=False, labelright=False, length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.set_size_inches(4.6, 2.4)
    fig.tight_layout(pad=0.4)
    fig.savefig(ASSETS_DIR / "opproplot_hero.png", dpi=220, transparent=True, bbox_inches="tight")
    plt.close(fig)


def generate_simulated_example() -> None:
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=5000)
    scores = rng.random(size=5000)

    fig, _, _ = operating_profile_plot(
        y_true,
        scores,
        bins=30,
        show_accuracy=True,
        show_key=True,
        key_location="inside",
        show_grid=False,
        title="Opproplot: Operating Profile",
    )
    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "opproplot_example.png", dpi=200)
    plt.close(fig)


def generate_breast_cancer() -> None:
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.25, random_state=0, stratify=data.target
    )
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_score = clf.predict_proba(X_test)[:, 1]

    fig, ax_hist, _ = operating_profile_plot(
        y_test,
        y_score,
        bins=30,
        show_accuracy=True,
        show_key=True,
        key_location="inside",
        show_grid=False,
        title="Breast cancer classifier: operating profile",
    )
    ax_hist.set_title("Breast cancer classifier: operating profile", fontsize=11)
    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "opproplot_breast_cancer.png", dpi=200)
    plt.close(fig)


def main() -> None:
    _ensure_assets_dir()
    generate_hero()
    generate_simulated_example()
    generate_breast_cancer()
    print("Generated docs images in docs/assets/")


if __name__ == "__main__":
    main()
