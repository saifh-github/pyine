"""
Utils:
- Filter experiments by metadata
- Load and aggregate multiple experiments into a single DataFrame
- Print summaries of the experiments
"""

import pathlib
import typing

import pandas as pd

import pyine.apps.code_execution_complexity.utils.persistence as persistence
import pyine.utils.code.complexity_metrics


def filter_experiments_by_metadata(
    experiments: list[persistence.ExperimentIdentifier],
    predictor_model: str | None = None,
    reasoning_effort: str | None = None,
    grader_model: str | None = None,
    seed: int | None = None,
) -> list[persistence.ExperimentIdentifier]:
    """Filter experiments by metadata.

    Args:
        experiments: List of experiments from list_experiments()
        predictor_model: Filter by predictor model name (e.g., "gpt-4o", "gpt-5")
        reasoning_effort: Filter by reasoning effort (e.g., "medium", None)
        grader_model: Filter by grader model name
        seed: Filter by random seed

    Returns:
        Filtered list of experiments.
    """
    filtered: list[persistence.ExperimentIdentifier] = []

    for exp in experiments:
        _, metadata = persistence.load_experiment_results(exp.path)

        if predictor_model is not None and metadata.get("predictor_model") != predictor_model:
            continue
        if reasoning_effort is not None and metadata.get("reasoning_effort") != reasoning_effort:
            continue
        if grader_model is not None and metadata.get("grader_model") != grader_model:
            continue
        if seed is not None and metadata.get("seed") != seed:
            continue

        filtered.append(exp)

    return filtered


def load_and_aggregate_experiments(
    experiment_paths: list[pathlib.Path],
) -> tuple[pd.DataFrame, list[dict[str, typing.Any]]]:
    """Loads and aggregates multiple experiments into a single DataFrame.

    Args:
        experiment_paths: List of paths to experiment directories.

    Returns:
        A tuple of (aggregated DataFrame, list of metadata dicts).
    """
    all_results: list[dict[str, typing.Any]] = []
    all_metadata: list[dict[str, typing.Any]] = []

    for exp_path in experiment_paths:
        results, metadata = persistence.load_experiment_results(exp_path)
        all_results.extend(results)
        all_metadata.append(metadata)

    df = pd.DataFrame(all_results)
    return df, all_metadata


def get_experiment_summary(
    experiments: list[persistence.ExperimentIdentifier],
    metadata_list: list[dict[str, typing.Any]],
) -> pd.DataFrame:
    """Create a summary table of the experiments as a pandas DataFrame."""
    summary_rows: list[dict[str, typing.Any]] = []

    for exp, metadata in zip(experiments, metadata_list, strict=False):
        summary_rows.append(
            {
                "name": exp.name,
                "timestamp": exp.timestamp,
                # Backward compatibility: old experiments use "offset", new use "start_position"
                "start_position": metadata.get("start_position", metadata.get("offset", 0)),
                "predictor_model": metadata.get("predictor_model"),
                "reasoning_effort": metadata.get("reasoning_effort"),
                # Backward compatibility: old experiments use "num_problems", new use "num_snippets"
                "num_snippets": metadata.get("num_snippets", metadata.get("num_problems")),
                "total_test_cases": metadata.get("total_test_cases"),
                "successful_evaluations": metadata.get("successful_evaluations"),
                "overall_accuracy": metadata.get("overall_accuracy"),
            }
        )

    return pd.DataFrame(summary_rows)


@typing.no_type_check
def print_overall_summary(df: pd.DataFrame, num_experiments: int) -> None:
    """Prints statistics for the aggregated results of the experiment."""
    print(f"Overall Statistics (aggregated across {num_experiments} experiment(s)):")
    print(f"Total test cases: {len(df)}")

    print("\nAccuracy metrics:")
    print(f"  Hard match:     {df['hard_match'].mean() * 100:.1f}%")
    print(f"  Soft match:     {df['soft_match'].mean() * 100:.1f}%")
    print(f"  LLM match:      {df['llm_match'].mean() * 100:.1f}%")
    print(f"  Combined:       {df['correct'].mean() * 100:.1f}%")

    print("\nComplexity metric ranges:")
    all_metrics = list(pyine.utils.code.complexity_metrics.COMPLEXITY_METRICS) + ["input_length"]
    for metric in all_metrics:
        if metric in df.columns:
            print(f"  {metric:30s} {df[metric].min():.1f} - {df[metric].max():.1f}")


@typing.no_type_check
def print_correlation_table(df: pd.DataFrame) -> None:
    """Prints correlation between complexity metrics and accuracy."""
    print("\nCorrelation between complexity metrics and accuracy:")
    print("=" * 60)

    all_metrics = list(pyine.utils.code.complexity_metrics.COMPLEXITY_METRICS) + ["input_length"]
    correlations: list[tuple[str, typing.Any]] = []
    for metric in all_metrics:
        if metric in df.columns:
            corr = df[metric].corr(df["correct"])
            correlations.append((metric, corr))

    correlations.sort(key=lambda x: x[1])
    for metric, corr in correlations:
        print(f"  {metric:30s} {corr:+.4f}")
