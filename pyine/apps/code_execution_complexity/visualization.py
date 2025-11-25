"""
This module comprises functions to visualize the results of the code execution / complexity analysis experiment.

CAREFUL: this is heavily Claude 4.5 Sonnet generated; I (Damiano) just quickly reviewed and slightly edited some stuff.
"""

import typing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pyine.utils.code.complexity_metrics

METRIC_LABELS = {
    "cyclomatic_complexity_avg": "Cyclomatic Complexity (Avg)",
    "cyclomatic_complexity_max": "Cyclomatic Complexity (Max)",
    "cyclomatic_complexity_sum": "Cyclomatic Complexity (Sum)",
    "loc": "Lines of Code (LOC)",
    "lloc": "Logical Lines of Code (LLOC)",
    "sloc": "Source Lines of Code (SLOC)",
    "comments": "Comment Lines",
    "multi": "Multiline Statements",
    "blank": "Blank Lines",
    "halstead_volume": "Halstead Volume",
    "halstead_difficulty": "Halstead Difficulty",
    "halstead_effort": "Halstead Effort",
    "maintainability_index": "Maintainability Index",
    "input_length": "Input Length (characters)",
}


@typing.no_type_check
def plot_accuracy_comparison(
    results: list[dict[str, typing.Any]],
) -> plt.Figure:
    """Draws plots comparing three measures of accuracy."""
    df = pd.DataFrame(results)

    hard_accuracy = df["hard_match"].mean() * 100
    soft_accuracy = df["soft_match"].mean() * 100
    llm_accuracy = df["llm_match"].mean() * 100
    combined_accuracy = df["correct"].mean() * 100

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    measures = ["Hard\nMatch", "Soft\nMatch", "LLM\nGrader", "Combined"]
    accuracies = [hard_accuracy, soft_accuracy, llm_accuracy, combined_accuracy]
    colors = ["#ff9999", "#99ccff", "#99ff99", "#ffcc99"]

    ax = axes[0]
    bars = ax.bar(measures, accuracies, color=colors, edgecolor="black", alpha=0.7)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Overall Accuracy by Measure", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(y=50, color="gray", linestyle="--", linewidth=1, alpha=0.5)

    for bar, acc in zip(bars, accuracies, strict=False):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 2,
            f"{acc:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    only_hard = ((df["hard_match"] == 1) & (df["soft_match"] == 0) & (df["llm_match"] == 0)).sum()
    only_soft = ((df["hard_match"] == 0) & (df["soft_match"] == 1) & (df["llm_match"] == 0)).sum()
    only_llm = ((df["hard_match"] == 0) & (df["soft_match"] == 0) & (df["llm_match"] == 1)).sum()
    hard_soft = ((df["hard_match"] == 1) & (df["soft_match"] == 1) & (df["llm_match"] == 0)).sum()
    hard_llm = ((df["hard_match"] == 1) & (df["soft_match"] == 0) & (df["llm_match"] == 1)).sum()
    soft_llm = ((df["hard_match"] == 0) & (df["soft_match"] == 1) & (df["llm_match"] == 1)).sum()
    all_three = ((df["hard_match"] == 1) & (df["soft_match"] == 1) & (df["llm_match"] == 1)).sum()
    none = ((df["hard_match"] == 0) & (df["soft_match"] == 0) & (df["llm_match"] == 0)).sum()

    ax = axes[1]
    categories = [
        "All 3\nAgree\n(Correct)",
        "Hard+Soft\nOnly",
        "Hard+LLM\nOnly",
        "Soft+LLM\nOnly",
        "Hard\nOnly",
        "Soft\nOnly",
        "LLM\nOnly",
        "All 3\nDisagree\n(Wrong)",
    ]
    counts = [all_three, hard_soft, hard_llm, soft_llm, only_hard, only_soft, only_llm, none]
    colors2 = ["green", "lightblue", "lightblue", "lightblue", "orange", "orange", "orange", "red"]

    bars2 = ax.bar(range(len(categories)), counts, color=colors2, edgecolor="black", alpha=0.7)
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylabel("Number of Test Cases", fontsize=12)
    ax.set_title("Agreement Between Measures", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    for bar, count in zip(bars2, counts, strict=False):
        if count > 0:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 1,
                f"{count}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    ax = axes[2]
    agreement_data = []
    for measure1, col1 in [("Hard", "hard_match"), ("Soft", "soft_match"), ("LLM", "llm_match")]:
        row = []
        for measure2, col2 in [("Hard", "hard_match"), ("Soft", "soft_match"), ("LLM", "llm_match")]:
            if measure1 == measure2:
                agreement = 100.0
            else:
                agreement = ((df[col1] == df[col2]).sum() / len(df)) * 100
            row.append(agreement)
        agreement_data.append(row)

    im = ax.imshow(agreement_data, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(["Hard", "Soft", "LLM"])
    ax.set_yticklabels(["Hard", "Soft", "LLM"])
    ax.set_title("Pairwise Agreement (%)", fontsize=14, fontweight="bold")

    for i in range(3):
        for j in range(3):
            ax.text(
                j,
                i,
                f"{agreement_data[i][j]:.1f}%",
                ha="center",
                va="center",
                color="black",
                fontsize=11,
                fontweight="bold",
            )

    plt.colorbar(im, ax=ax, label="Agreement %")
    plt.tight_layout()

    return fig


@typing.no_type_check
def plot_complexity_distributions(
    results: list[dict[str, typing.Any]],
    metrics: list[tuple[str, str, bool]] | None = None,
) -> plt.Figure:
    """For each complexity measure, plots histograms showing how many code snippets have a specific
    value of that complexity measure.
    """
    if metrics is None:
        metrics = [
            (metric, METRIC_LABELS.get(metric, metric.replace("_", " ").title()), False)
            for metric in pyine.utils.code.complexity_metrics.COMPLEXITY_METRICS
        ]
        metrics.append(("input_length", METRIC_LABELS.get("input_length", "Input Length"), True))

    df = pd.DataFrame(results)
    df_unique = df.drop_duplicates(subset="problem_id")

    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    axes = axes.flatten()

    for idx, (metric_col, metric_label, is_per_test) in enumerate(metrics):
        ax = axes[idx]

        df_to_use = df if is_per_test else df_unique
        values = df_to_use[metric_col].dropna()

        ax.hist(values, bins=10, edgecolor="black", alpha=0.7)
        ax.set_title(metric_label, fontsize=12, fontweight="bold")
        ax.set_xlabel("Value", fontsize=10)

        ylabel = "Frequency (# of test cases)" if is_per_test else "Frequency (# of code snippets)"
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(axis="y", alpha=0.3)

        mean_val = values.mean()
        median_val = values.median()
        ax.axvline(mean_val, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_val:.2f}")
        ax.axvline(median_val, color="green", linestyle="--", linewidth=2, label=f"Median: {median_val:.2f}")
        ax.legend(fontsize=8)

    for idx in range(len(metrics), len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    return fig


@typing.no_type_check
def plot_correlation_analysis(
    results: list[dict[str, typing.Any]],
) -> plt.Figure:
    """Draws plots of the correlation between a given complexity metrics and the LLM accuracy."""
    df = pd.DataFrame(results)

    complexity_metrics = list(pyine.utils.code.complexity_metrics.COMPLEXITY_METRICS)
    complexity_metrics.append("input_length")

    correlations = {}
    for metric in complexity_metrics:
        # check for zero variance before computing correlation
        if df[metric].std() == 0:
            correlations[metric] = 0.0
        else:
            corr = df[metric].corr(df["correct"])
            correlations[metric] = 0.0 if pd.isna(corr) else corr

    corr_df = pd.DataFrame(
        [{"Metric": metric, "Correlation": corr} for metric, corr in correlations.items()]
    ).sort_values("Correlation")

    # adjust figure height based on number of metrics
    fig_height = max(6, len(corr_df) * 0.6)
    fig, ax = plt.subplots(1, 1, figsize=(10, fig_height))

    colors = ["red" if c < 0 else "green" for c in corr_df["Correlation"]]
    ax.barh(corr_df["Metric"], corr_df["Correlation"], color=colors, alpha=0.7, edgecolor="black")
    ax.set_xlabel("Correlation with Accuracy", fontsize=12)
    ax.set_title("Complexity Metrics vs Accuracy Correlation", fontsize=14, fontweight="bold")
    ax.axvline(x=0, color="black", linestyle="-", linewidth=1)
    ax.grid(axis="x", alpha=0.3)

    for i, (_metric, corr) in enumerate(zip(corr_df["Metric"], corr_df["Correlation"], strict=False)):
        ax.text(
            corr + 0.01 if corr > 0 else corr - 0.01,
            i,
            f"{corr:.3f}",
            ha="left" if corr > 0 else "right",
            va="center",
            fontsize=9,
        )

    plt.tight_layout()
    return fig


@typing.no_type_check
def plot_accuracy_vs_complexity(
    results: list[dict[str, typing.Any]],
    metric: str,
    metric_label: str,
    num_bins: int = 5,
) -> plt.Figure:
    """For a given complexity metric, considers a numerical range for that metric, and plots the
    accuracy of the LLM in that range.
    """
    df = pd.DataFrame(results)
    df["bin"] = pd.qcut(df[metric], q=num_bins, duplicates="drop")

    bin_stats = df.groupby("bin", observed=True).agg({"correct": ["mean", "count"]}).reset_index()
    bin_stats.columns = ["bin", "accuracy", "count"]

    fig, ax = plt.subplots(1, 1, figsize=(14, 6))

    x_pos = np.arange(len(bin_stats))
    bars = ax.bar(x_pos, bin_stats["accuracy"] * 100, alpha=0.7, edgecolor="black")

    for bar, acc in zip(bars, bin_stats["accuracy"], strict=False):
        if acc >= 0.8:
            bar.set_color("green")
        elif acc >= 0.5:
            bar.set_color("orange")
        else:
            bar.set_color("red")

    for i, (acc, count) in enumerate(zip(bin_stats["accuracy"], bin_stats["count"], strict=False)):
        ax.text(i, acc * 100 + 2, f"n={int(count)}", ha="center", fontsize=8)

    ax.set_title(f"Accuracy vs {metric_label}", fontsize=14, fontweight="bold")
    ax.set_xlabel(f"{metric_label} Range", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(b) for b in bin_stats["bin"]], rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(y=50, color="gray", linestyle="--", linewidth=1, alpha=0.5)

    plt.tight_layout()
    return fig


@typing.no_type_check
def plot_all_accuracy_vs_complexity(
    results: list[dict[str, typing.Any]],
    num_bins: int = 5,
) -> list[plt.Figure]:
    """For each complexity metric in a list, considers a numerical range for that metric, and plots
    the accuracy of the LLM in that range.
    """
    df = pd.DataFrame(results)
    all_metrics = list(pyine.utils.code.complexity_metrics.COMPLEXITY_METRICS) + ['input_length']
    figures = []

    for metric in all_metrics:
        if metric in df.columns:
            label = METRIC_LABELS.get(metric, metric.replace("_", " ").title())
            fig = plot_accuracy_vs_complexity(results, metric, label, num_bins)
            figures.append(fig)

    return figures
