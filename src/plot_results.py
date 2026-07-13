#!/usr/bin/env python
"""Plot success and utility rates per model × system prompt from all_counts JSON."""

import sys
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from config import OUTPUT_FP

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = Path(str(OUTPUT_FP).replace(".jsonl", "_all_counts.json"))
DEFAULT_OUTPUT_DIR = ROOT_DIR / "results"
DESC_KEYS = ("short", "full")


def _collect(all_counts, metric: str, condition: str) -> tuple[list, list, dict]:
    """Return (models, prompt_indices, data) where data[model][(prompt_idx, desc_key)] = rate."""
    models = list(all_counts.keys())
    prompt_indices = sorted({idx for prompts in all_counts.values() for idx in prompts})
    data = {model: {} for model in models}
    for model_name, prompts in all_counts.items():
        for prompt_idx, desc_modes in prompts.items():
            for desc_key in DESC_KEYS:
                if desc_key in desc_modes and condition in desc_modes[desc_key]:
                    data[model_name][(prompt_idx, desc_key)] = (
                        desc_modes[desc_key][condition].get(metric, 0.0)
                    )
    return models, prompt_indices, data


def _plot(models: list, prompt_indices: list, data: dict,
          title: str, ylabel: str, output_path: Path) -> None:
    groups = [(idx, dk) for idx in prompt_indices for dk in DESC_KEYS]
    n_models = len(models)
    n_groups = len(groups)
    bar_width = 0.8 / n_models
    x = np.arange(n_groups)
    offsets = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * bar_width
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, ax = plt.subplots(figsize=(max(6, n_groups * n_models * 1.0), 5))
    for i, model_name in enumerate(models):
        rates = [data[model_name].get(g, 0.0) * 100 for g in groups]
        bars = ax.bar(x + offsets[i], rates, width=bar_width,
                      label=model_name, color=colors[i % len(colors)], edgecolor="white")
        ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([f"prompt {idx}\n{dk}" for idx, dk in groups], fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, 100)
    ax.legend(title="Model")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Bar charts of success and utility rates by model × system prompt."
    )
    parser.add_argument(
        '--input-fp',
        type=Path,
        default=DEFAULT_INPUT,
        metavar='PATH',
        help=f'all_counts JSON file (default: {DEFAULT_INPUT})',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        metavar='PATH',
        help=f'Directory to save plots (default: {DEFAULT_OUTPUT_DIR})',
    )
    args = parser.parse_args()

    with open(args.input_fp) as f:
        all_counts = json.load(f)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    charts = [
        ("success", "poison", "Adversary success rate — poisoned tools", "Success rate (%)"),
        ("ignored", "poison", "Utility rate (correct tool called) — poisoned tools", "Utility rate (%)"),
        ("ignored", "benign", "Utility rate (correct tool called) — benign tools", "Utility rate (%)"),
    ]
    filenames = ["success_rate_poison.png", "utility_rate_poison.png", "utility_rate_benign.png"]

    for (metric, condition, title, ylabel), filename in zip(charts, filenames):
        models, prompt_indices, data = _collect(all_counts, metric, condition)
        _plot(models, prompt_indices, data, title, ylabel, args.output_dir / filename)


if __name__ == "__main__":
    main()
