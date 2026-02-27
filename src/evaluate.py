"""
Evaluation script for aggregating metrics and generating comparison visualizations.
Fetches results from WandB and creates comparative plots.
"""

import os
import sys
import json
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from typing import List, Dict, Any
import wandb
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def fetch_run_data(entity: str, project: str, run_id: str) -> Dict[str, Any]:
    """
    Fetch run data from WandB API.

    Args:
        entity: WandB entity
        project: WandB project
        run_id: Run display name

    Returns:
        Dictionary with run config, summary, and history
    """
    api = wandb.Api()

    # Search for runs with this display name
    runs = api.runs(
        f"{entity}/{project}", filters={"display_name": run_id}, order="-created_at"
    )

    if not runs:
        raise ValueError(
            f"No runs found with display_name='{run_id}' in {entity}/{project}"
        )

    # Get most recent run
    run = runs[0]

    # Extract data
    config = dict(run.config)
    summary = dict(run.summary)

    # Get history (if available)
    history = []
    try:
        for row in run.scan_history():
            history.append(dict(row))
    except Exception as e:
        print(f"Warning: Could not fetch history for {run_id}: {e}")

    return {"config": config, "summary": summary, "history": history, "url": run.url}


def bootstrap_ci(
    data: List[float], confidence: float = 0.95, n_bootstrap: int = 1000
) -> tuple:
    """
    Calculate bootstrap confidence interval.

    Args:
        data: List of values
        confidence: Confidence level
        n_bootstrap: Number of bootstrap samples

    Returns:
        (mean, lower_bound, upper_bound)
    """
    if not data:
        return (0.0, 0.0, 0.0)

    data = np.array(data)
    means = []

    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        means.append(np.mean(sample))

    mean = np.mean(data)
    alpha = 1 - confidence
    lower = np.percentile(means, 100 * alpha / 2)
    upper = np.percentile(means, 100 * (1 - alpha / 2))

    return (mean, lower, upper)


def create_comparison_plots(
    run_data: Dict[str, Dict[str, Any]], output_dir: Path
) -> List[str]:
    """
    Create comparison plots for all runs.

    Args:
        run_data: Dictionary mapping run_id to run data
        output_dir: Directory to save plots

    Returns:
        List of generated file paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files = []

    # Extract metrics from all runs
    run_ids = list(run_data.keys())
    metrics = {}

    for run_id, data in run_data.items():
        summary = data["summary"]
        for key, value in summary.items():
            if isinstance(value, (int, float)):
                if key not in metrics:
                    metrics[key] = {}
                metrics[key][run_id] = value

    # 1. Accuracy comparison bar chart
    if "accuracy" in metrics:
        fig, ax = plt.subplots(figsize=(10, 6))

        run_ids_sorted = sorted(metrics["accuracy"].keys())
        accuracies = [metrics["accuracy"][rid] for rid in run_ids_sorted]

        colors = [
            "#2ecc71" if "proposed" in rid else "#3498db" for rid in run_ids_sorted
        ]
        bars = ax.bar(range(len(run_ids_sorted)), accuracies, color=colors, alpha=0.8)

        ax.set_xlabel("Method", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_title(
            "Accuracy Comparison Across Methods", fontsize=14, fontweight="bold"
        )
        ax.set_xticks(range(len(run_ids_sorted)))
        ax.set_xticklabels(run_ids_sorted, rotation=45, ha="right")
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.3)

        # Add value labels on bars
        for i, (bar, acc) in enumerate(zip(bars, accuracies)):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{acc:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        plt.tight_layout()
        filepath = output_dir / "comparison_accuracy.pdf"
        plt.savefig(filepath, format="pdf", bbox_inches="tight")
        plt.close()
        generated_files.append(str(filepath))
        print(f"Generated: {filepath}")

    # 2. Average calls comparison
    if "avg_calls" in metrics:
        fig, ax = plt.subplots(figsize=(10, 6))

        run_ids_sorted = sorted(metrics["avg_calls"].keys())
        avg_calls = [metrics["avg_calls"][rid] for rid in run_ids_sorted]

        colors = [
            "#2ecc71" if "proposed" in rid else "#3498db" for rid in run_ids_sorted
        ]
        bars = ax.bar(range(len(run_ids_sorted)), avg_calls, color=colors, alpha=0.8)

        ax.set_xlabel("Method", fontsize=12)
        ax.set_ylabel("Average API Calls", fontsize=12)
        ax.set_title("Average API Calls per Question", fontsize=14, fontweight="bold")
        ax.set_xticks(range(len(run_ids_sorted)))
        ax.set_xticklabels(run_ids_sorted, rotation=45, ha="right")
        ax.grid(axis="y", alpha=0.3)

        # Add value labels on bars
        for i, (bar, calls) in enumerate(zip(bars, avg_calls)):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.05,
                f"{calls:.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        plt.tight_layout()
        filepath = output_dir / "comparison_avg_calls.pdf"
        plt.savefig(filepath, format="pdf", bbox_inches="tight")
        plt.close()
        generated_files.append(str(filepath))
        print(f"Generated: {filepath}")

    # 3. Accuracy vs Cost scatter plot
    if "accuracy" in metrics and "avg_calls" in metrics:
        fig, ax = plt.subplots(figsize=(10, 6))

        for run_id in run_ids:
            acc = metrics["accuracy"].get(run_id, 0)
            calls = metrics["avg_calls"].get(run_id, 0)

            color = "#2ecc71" if "proposed" in run_id else "#3498db"
            marker = "o" if "proposed" in run_id else "s"
            size = 150 if "proposed" in run_id else 100

            ax.scatter(
                calls,
                acc,
                s=size,
                c=color,
                marker=marker,
                alpha=0.7,
                edgecolors="black",
                linewidth=1.5,
                label=run_id,
            )

        ax.set_xlabel("Average API Calls per Question", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_title(
            "Accuracy vs. API Call Budget (Pareto Frontier)",
            fontsize=14,
            fontweight="bold",
        )
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", framealpha=0.9)
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1.0)

        plt.tight_layout()
        filepath = output_dir / "comparison_pareto.pdf"
        plt.savefig(filepath, format="pdf", bbox_inches="tight")
        plt.close()
        generated_files.append(str(filepath))
        print(f"Generated: {filepath}")

    return generated_files


# Register default config with ConfigStore
cs = ConfigStore.instance()
cs.store(name="eval_config", node={"results_dir": ".research/results", "run_ids": "[]"})


@hydra.main(version_base=None, config_path=None, config_name="eval_config")
def main(cfg: DictConfig):
    """Main evaluation function."""
    # [VALIDATOR FIX - Attempt 1]
    # [PROBLEM]: argparse expects --results_dir and --run_ids but workflow passes results_dir= and run_ids= (Hydra style)
    # [CAUSE]: evaluate.py used argparse while workflow called it with Hydra-style arguments
    # [FIX]: Converted from argparse to Hydra with a default config to accept CLI overrides without + prefix
    #
    # [OLD CODE]:
    # parser = argparse.ArgumentParser(description="Evaluate and compare experiment runs")
    # parser.add_argument(
    #     "--results_dir", type=str, required=True, help="Results directory"
    # )
    # parser.add_argument(
    #     "--run_ids", type=str, required=True, help="JSON string list of run IDs"
    # )
    # args = parser.parse_args()
    # run_ids = json.loads(args.run_ids)
    # results_dir = Path(args.results_dir)
    #
    # [NEW CODE]:

    # Parse run_ids from Hydra config (passed as CLI overrides)
    # Hydra may parse JSON arrays directly into ListConfig, so handle both cases
    if isinstance(cfg.run_ids, str):
        run_ids = json.loads(cfg.run_ids)
    else:
        # Already parsed by Hydra as a list
        run_ids = list(cfg.run_ids)

    results_dir = Path(cfg.results_dir)

    print(f"Evaluating {len(run_ids)} runs:")
    for run_id in run_ids:
        print(f"  - {run_id}")

    # Get WandB config from environment
    entity = os.environ.get("WANDB_ENTITY", "airas")
    project = os.environ.get("WANDB_PROJECT", "2026-0227-matsuzawa")

    print(f"\nFetching data from WandB ({entity}/{project})...")

    # Fetch data for all runs
    run_data = {}
    for run_id in run_ids:
        print(f"Fetching: {run_id}")
        try:
            data = fetch_run_data(entity, project, run_id)
            run_data[run_id] = data
            print(f"  ✓ Found run: {data['url']}")
        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue

    if not run_data:
        print("Error: No run data found!")
        sys.exit(1)

    # Export per-run metrics
    print("\nExporting per-run metrics...")
    for run_id, data in run_data.items():
        run_dir = results_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        metrics_file = run_dir / "metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(data["summary"], f, indent=2)
        print(f"Saved: {metrics_file}")

    # Aggregate metrics
    print("\nAggregating metrics...")
    comparison_dir = results_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    # Extract key metrics
    aggregated = {"primary_metric": "accuracy", "metrics_by_run": {}}

    for run_id, data in run_data.items():
        aggregated["metrics_by_run"][run_id] = data["summary"]

    # Calculate best proposed and baseline
    proposed_runs = {k: v for k, v in run_data.items() if "proposed" in k}
    baseline_runs = {k: v for k, v in run_data.items() if "comparative" in k}

    if proposed_runs:
        best_proposed = max(
            proposed_runs.items(), key=lambda x: x[1]["summary"].get("accuracy", 0)
        )
        aggregated["best_proposed"] = {
            "run_id": best_proposed[0],
            "accuracy": best_proposed[1]["summary"].get("accuracy", 0),
        }

    if baseline_runs:
        best_baseline = max(
            baseline_runs.items(), key=lambda x: x[1]["summary"].get("accuracy", 0)
        )
        aggregated["best_baseline"] = {
            "run_id": best_baseline[0],
            "accuracy": best_baseline[1]["summary"].get("accuracy", 0),
        }

    if proposed_runs and baseline_runs:
        gap = (
            aggregated["best_proposed"]["accuracy"]
            - aggregated["best_baseline"]["accuracy"]
        )
        aggregated["gap"] = gap
        print(f"Performance gap: {gap:+.4f}")

    # Save aggregated metrics
    agg_file = comparison_dir / "aggregated_metrics.json"
    with open(agg_file, "w") as f:
        json.dump(aggregated, f, indent=2)
    print(f"Saved: {agg_file}")

    # Generate comparison plots
    print("\nGenerating comparison plots...")
    plot_files = create_comparison_plots(run_data, comparison_dir)

    print("\n" + "=" * 80)
    print("Evaluation complete!")
    print("=" * 80)
    print(f"\nGenerated files:")
    print(f"  - {agg_file}")
    for pf in plot_files:
        print(f"  - {pf}")


if __name__ == "__main__":
    main()
