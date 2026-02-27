"""
Main orchestrator for inference experiments.
Handles configuration and mode overrides.
"""

import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

from src.inference import run_inference


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for experiment execution.

    Args:
        cfg: Hydra configuration
    """
    print("=" * 80)
    print(f"Running experiment: {cfg.run.run_id}")
    print(f"Mode: {cfg.mode}")
    print("=" * 80)

    # Apply mode-specific overrides
    if cfg.mode == "sanity_check":
        print("\nApplying sanity_check mode overrides:")

        # Reduce dataset size for sanity check
        original_samples = cfg.dataset.num_samples
        cfg.dataset.num_samples = min(10, original_samples)
        print(
            f"  - dataset.num_samples: {original_samples} -> {cfg.dataset.num_samples}"
        )

        # Keep wandb online for sanity checks
        print(f"  - wandb.mode: {cfg.wandb.mode} (keeping as-is)")

    # Print final configuration
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

    # Run inference
    run_inference(cfg)

    print("=" * 80)
    print("Experiment complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
