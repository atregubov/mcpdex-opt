#!/usr/bin/env python
"""Plot success and utility rates per model × system prompt from all_counts JSON."""

import sys
import argparse
import json
import math
import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from config import OUTPUT_FP

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = Path(str(OUTPUT_FP).replace(".jsonl", "_all_counts.json"))
DEFAULT_OPTIMIZATION_INPUT = Path(str(OUTPUT_FP).replace(".jsonl", "_optimization_all_counts.json"))
DEFAULT_OUTPUT_DIR = ROOT_DIR / "plots"
DESC_KEYS = ("short", "full")
DEFAULT_DESC_KEY = "full"
HEATMAP_DESC_ORDER = ("full", "short")
PLOT_BAR_CHARTS = False # useful only for small number of models

# Display order for prompt indices on the heatmap's horizontal axis.
HEATMAP_PROMPT_ORDER = ("0", "2", "4", "5", "1", "3", "6")

# Models excluded from the benign-tools utility heatmap.
HEATMAP_BENIGN_EXCLUDED_MODELS = []

# Models and system-prompt indices dropped from `all_counts` before any chart is
# built (see `_filter_all_counts`). Prompt indices are strings, matching JSON keys.
EXCLUDED_MODELS = []
EXCLUDED_PROMPTS = ["0", "2", "4"] # baseline and interventions
# EXCLUDED_PROMPTS = ["1", "3", "6"] # only baselines

# Prompt aliases/display order for the optimization-results heatmap: the optimization
# experiments use a different (smaller) prompt set than the main experiments.
OPTIMIZATION_PROMPT_ALIASES = {
    "0": "No-other-calls.Warning",
    "1": "Don't-help.No-calls",
    "2": "Baseline",
}
OPTIMIZATION_PROMPT_ORDER = ("1", "2", "0")

# Marker shapes cycled across prompts on the utility-vs-success scatter plots.
SCATTER_PROMPT_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")
SCATTER_DESC_COLORS = {"full": "tab:red", "short": "tab:blue"}

# Each experiment's per-step "success" value is itself an average over this many
# repeated agent runs (matches OPTIMIZATION_NUM_REPEATS in config.py at the time the
# optimization data was generated), so the true trial count backing a step's
# cross-experiment average is n_experiments * TRIALS_PER_EXPERIMENT_STEP.
TRIALS_PER_EXPERIMENT_STEP = 10

# Fixed number of experiments backing each point on the utility-vs-success scatter plot.
SCATTER_N_TRIALS = 423

# System-prompt index -> short display label, for heatmap axis labeling.
PROMPT_ALIASES = {
    "0": "Empty\nalternative baseline",
    "1": "No-other-calls.Caution",
    "2": "Trust-no-one.No-calls\nalternative baseline",
    "3": "No-other-calls.Warning",
    "4": "Don't-help.No-calls\nalternative baseline",
    "5": "Baseline",
    "6": "Optimized defense",
} # order: "0", "2", "4", "5", "1", "3", "6"

# Model name -> short display label, for heatmap axis labeling.
MODEL_ALIASES = {
    "gpt-5.4": "GPT-5.4",
    "gpt-4.1": "GPT-4.1",
    "gemini-3.5-flash": "Gemini-3.5-flash",
    "deepseek.v3.2": "Deepseek.v3.2",
    "minimax.minimax-m2.5": "Minimax-m2.5",
    "mistral.devstral-2-123b": "Devstral-2-123b",
    "mistral.ministral-3-8b-instruct": "Ministral-3-8b",
    "mistral.mistral-large-3-675b-instruct": "Mistral-large-3-675b",
    "moonshotai.kimi-k2.5": "Kimi-k2.5",
    "openai.gpt-oss-safeguard-120b": "GPT-OSS-safeguard-120b",
    "openai.gpt-oss-safeguard-20b": "GPT-OSS-safeguard-20b",
    "openai.gpt-oss-20b": "GPT-OSS-20b",
    "openai.gpt-oss-120b": "GPT-OSS-120b",
    "qwen.qwen3-32b": "Qwen3-32b",
    "qwen.qwen3-235b-a22b-2507": "Qwen3-235b",
    "qwen.qwen3-coder-next": "Qwen3-coder",
    "nvidia.nemotron-super-3-120b": "Nemotron-super-3-120b",
    "nvidia.nemotron-nano-3-30b": "Nemotron-nano-3-30b",
    "zai.glm-5": "Zai.glm-5",
}


def _binomial_se(p: float, n: int) -> float:
    """Return the standard error of a proportion ``p`` estimated from ``n`` trials."""
    return math.sqrt(p * (1 - p) / n) if n > 0 else 0.0


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
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved to {output_path}")


def _collect_heatmap(all_counts: dict, metric: str, condition: str,
                     prompt_order: tuple = HEATMAP_PROMPT_ORDER,
                     summary_key: str | None = None) -> tuple[list, list, np.ndarray]:
    """Return (models, columns, matrix) for a model x (prompt, desc_key) heatmap.

    ``columns`` lists every ``(prompt_idx, desc_key)`` pair present in ``all_counts``,
    grouped by ``HEATMAP_DESC_ORDER`` ("full" before "short") - i.e. every prompt's
    "full" column appears before any prompt's "short" column - and ordered by
    ``prompt_order`` within each group (any prompt index not listed there is appended
    afterward, sorted numerically). ``matrix[i, j]`` is the ``metric`` rate (0.0-1.0)
    for ``models[i]`` under ``condition`` at ``columns[j]``, or ``0.0`` when missing.

    Args:
        summary_key: When set (e.g. ``"summary"``), ``metric`` is looked up one level
            deeper, under ``desc_modes[desc_key][condition][summary_key]`` - for
            reading optimization-results JSON, where each condition holds a
            ``{"summary": {...}, "data": {...}}`` pair rather than the rate dict
            directly.
    """
    models = list(all_counts.keys())
    all_prompt_idx = {idx for prompts in all_counts.values() for idx in prompts}
    leftover = sorted(all_prompt_idx - set(prompt_order), key=int)
    prompt_indices = [idx for idx in prompt_order if idx in all_prompt_idx] + leftover
    columns = [(idx, dk) for dk in HEATMAP_DESC_ORDER for idx in prompt_indices]

    matrix = np.zeros((len(models), len(columns)))
    for i, model_name in enumerate(models):
        for j, (prompt_idx, desc_key) in enumerate(columns):
            desc_modes = all_counts[model_name].get(prompt_idx, {})
            condition_data = desc_modes.get(desc_key, {}).get(condition, {})
            if summary_key is not None:
                condition_data = condition_data.get(summary_key, {})
            matrix[i, j] = condition_data.get(metric, 0.0)

    return models, columns, matrix


def _column_label(prompt_idx: str, desc_key: str, prompt_aliases: dict = None) -> str:
    """Build a heatmap column label: the prompt alias, suffixed with ``*`` for ``short``."""
    prompt_aliases = PROMPT_ALIASES if prompt_aliases is None else prompt_aliases
    alias = prompt_aliases.get(prompt_idx, prompt_idx)
    return f"{alias}*\nwith shortening" if desc_key == "short" else alias


def _plot_heatmap(models: list, columns: list, matrix: np.ndarray,
                  title: str, cbar_label: str, output_path: Path,
                  prompt_aliases: dict = None, cmap: str = "Reds",
                  vmin: float = 0, vmax: float = 100, signed: bool = False) -> None:
    """Render a model x (prompt, desc_key) heatmap of ``matrix``.

    By default renders rates (``matrix`` in 0.0-1.0) on a sequential 0-100 scale.
    Pass ``cmap="RdBu_r", vmin=-100, vmax=100, signed=True`` to render a difference
    heatmap instead (``matrix`` in -1.0-1.0), which uses a diverging colormap and
    signed (``+``/``-``) cell labels.
    """
    row_labels = [MODEL_ALIASES.get(model_name, model_name) for model_name in models]
    col_labels = [_column_label(idx, dk, prompt_aliases) for idx, dk in columns]
    values = matrix * 100

    fig, ax = plt.subplots(figsize=(max(6, len(col_labels) * 0.9), max(4, len(row_labels) * 0.5)))
    im = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)

    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            text = f"{values[i, j]:+.0f}" if signed else f"{values[i, j]:.0f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=8,
                    color="white" if abs(values[i, j]) > 50 else "black")

    ax.set_title(title, fontsize=10, wrap=True)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved to {output_path}")


def plot_success_heatmap(all_counts: dict, output_path: Path) -> None:
    """Plot a model x (prompt, desc_key) heatmap of adversary success rate.

    Only the ``benign_poison_key_ == "poison"`` condition is plotted - the "success"
    outcome (adversarial tool called) is only meaningful when the poisoned tool is
    actually served.

    Args:
        all_counts: Aggregate counts dict as produced by ``run_batch.py``
            (``model -> system_prompt_idx -> short/full -> benign/poison -> label -> rate``).
        output_path: File path to save the resulting PNG to.
    """
    models, columns, matrix = _collect_heatmap(all_counts, metric="success", condition="poison")
    _plot_heatmap(models, columns, matrix,
                 title="Attack Success Rate (with poisoned tools).\n'*' - shortening and rephrasing applied",
                 cbar_label="Attack Success Rate (%, lower is better)",
                 output_path=output_path)


def plot_utility_heatmap(all_counts: dict, output_path: Path) -> None:
    """Plot a model x (prompt, desc_key) heatmap of utility rate (correct tool called).

    Same structure as :func:`plot_success_heatmap` (models on the vertical axis,
    prompt x description-mode pairs on the horizontal axis, grouped "full" then
    "short"), but for the ``"ignored"`` metric, restricted to ``benign_poison_key_ ==
    "poison"`` - i.e. how often the model calls the correct tool and ignores the
    poisoned one, when the poisoned tool is present.

    Args:
        all_counts: Aggregate counts dict as produced by ``run_batch.py``
            (``model -> system_prompt_idx -> short/full -> benign/poison -> label -> rate``).
        output_path: File path to save the resulting PNG to.
    """
    models, columns, matrix = _collect_heatmap(all_counts, metric="ignored", condition="poison")
    _plot_heatmap(models, columns, matrix,
                 title="Utility rate (with poisoned tools)\n'*' - shortening and rephrasing applied",
                 cbar_label="Utility rate (%, higher is better)",
                 output_path=output_path)


def plot_utility_heatmap_benign(all_counts: dict, output_path: Path) -> None:
    """Plot a model x (prompt, desc_key) heatmap of utility rate for benign-only runs.

    Same structure as :func:`plot_utility_heatmap`, but restricted to
    ``benign_poison_key_ == "benign"`` (no poisoned tool served), and excludes
    ``HEATMAP_BENIGN_EXCLUDED_MODELS`` from the plotted rows.

    Args:
        all_counts: Aggregate counts dict as produced by ``run_batch.py``
            (``model -> system_prompt_idx -> short/full -> benign/poison -> label -> rate``).
        output_path: File path to save the resulting PNG to.
    """
    filtered_counts = {model_name: prompts for model_name, prompts in all_counts.items()
                       if model_name not in HEATMAP_BENIGN_EXCLUDED_MODELS}
    models, columns, matrix = _collect_heatmap(filtered_counts, metric="ignored", condition="benign")
    _plot_heatmap(models, columns, matrix,
                 title="Utility rate (benign only tools).\n'*' - shortening and rephrasing applied",
                 cbar_label="Utility rate (%, higher is better)",
                 output_path=output_path)


def _find_prompt_idx_by_alias(alias_name: str, prompt_aliases: dict = None) -> str:
    """Return the prompt index whose alias in ``prompt_aliases`` equals ``alias_name``.

    Raises:
        ValueError: If no prompt index maps to ``alias_name``.
    """
    prompt_aliases = PROMPT_ALIASES if prompt_aliases is None else prompt_aliases
    for idx, alias in prompt_aliases.items():
        if alias == alias_name:
            return idx
    raise ValueError(f"No prompt index found with alias {alias_name!r} in {prompt_aliases}")


def _collect_heatmap_vs_baseline(all_counts: dict, metric: str, condition: str,
                                 baseline_alias: str = "Baseline",
                                 prompt_order: tuple = HEATMAP_PROMPT_ORDER,
                                 prompt_aliases: dict = None) -> tuple[list, list, np.ndarray]:
    """Return (models, columns, matrix) of rates expressed relative to a baseline prompt,
    for a model x (prompt, desc_key) heatmap.

    Same models/columns as :func:`_collect_heatmap`, but each cell is
    ``rate(prompt, desc_key) - rate(baseline_prompt, desc_key)`` — every prompt's
    rate is compared against the ``baseline_alias`` prompt (default "Baseline")
    within the *same* description mode (full vs. full-baseline, short vs.
    short-baseline). The baseline prompt's own column becomes 0 (self-difference).

    Raises:
        ValueError: If no prompt index maps to ``baseline_alias`` (see
            :func:`_find_prompt_idx_by_alias`).
    """
    prompt_aliases = PROMPT_ALIASES if prompt_aliases is None else prompt_aliases
    baseline_idx = _find_prompt_idx_by_alias(baseline_alias, prompt_aliases)

    models, columns, matrix = _collect_heatmap(all_counts, metric, condition, prompt_order=prompt_order)
    diff_matrix = np.zeros_like(matrix)

    baseline_col = columns.index((baseline_idx, DEFAULT_DESC_KEY)) # only use "Baseline" from "full" description
    for j, _ in enumerate(columns):
        diff_matrix[:, j] = matrix[:, j] - matrix[:, baseline_col]

    return models, columns, diff_matrix


def plot_success_heatmap_vs_default(all_counts: dict, output_path: Path) -> None:
    """Plot a model x (prompt, desc_key) heatmap of adversary success rate relative to
    the "Baseline" system prompt.

    Same data/condition as :func:`plot_success_heatmap`, but every column's value is
    ``rate - rate(Baseline prompt, same description mode)`` instead of the raw rate —
    i.e. how much more (red) or less (blue) successful the attack was under this
    prompt compared to the default one. The "Baseline" columns themselves are always 0.

    Args:
        all_counts: Aggregate counts dict as produced by ``run_batch.py``
            (``model -> system_prompt_idx -> short/full -> benign/poison -> label -> rate``).
        output_path: File path to save the resulting PNG to.
    """
    models, columns, matrix = _collect_heatmap_vs_baseline(all_counts, metric="success", condition="poison")
    _plot_heatmap(models, columns, matrix,
                 title="Attack Success Rate relative to 'Baseline'(with poisoned tools)\n"
                       "'*' - shortening and rephrasing applied",
                 cbar_label="Attack Success Rate change (pp), (+) ASR increase, (-) ASR decrease, lower is better",
                 output_path=output_path,
                 cmap="RdBu_r", vmin=-100, vmax=100, signed=True)


def plot_utility_heatmap_vs_default(all_counts: dict, output_path: Path) -> None:
    """Plot a model x (prompt, desc_key) heatmap of utility rate relative to the
    "Baseline" system prompt.

    Same data/condition as :func:`plot_utility_heatmap`, but every column's value is
    ``rate - rate(Baseline prompt, same description mode)`` instead of the raw rate.
    The "Baseline" columns themselves are always 0.

    Args:
        all_counts: Aggregate counts dict as produced by ``run_batch.py``
            (``model -> system_prompt_idx -> short/full -> benign/poison -> label -> rate``).
        output_path: File path to save the resulting PNG to.
    """
    models, columns, matrix = _collect_heatmap_vs_baseline(all_counts, metric="ignored", condition="poison")
    _plot_heatmap(models, columns, matrix,
                 title="Utility rate change relative to 'Baseline' (with poisoned MCP tools)",
                 cbar_label="Utility rate change: (+) utility increase (higher is better), (-) utility decrease",
                 output_path=output_path,
                 cmap="RdBu_r", vmin=-100, vmax=100, signed=True)


def plot_utility_heatmap_benign_vs_default(all_counts: dict, output_path: Path) -> None:
    """Plot a model x (prompt, desc_key) heatmap of benign-tools utility rate relative
    to the "Baseline" system prompt.

    Same data/condition as :func:`plot_utility_heatmap_benign` (including the
    ``HEATMAP_BENIGN_EXCLUDED_MODELS`` filtering), but every column's value is
    ``rate - rate(Baseline prompt, same description mode)`` instead of the raw rate.
    The "Baseline" columns themselves are always 0.

    Args:
        all_counts: Aggregate counts dict as produced by ``run_batch.py``
            (``model -> system_prompt_idx -> short/full -> benign/poison -> label -> rate``).
        output_path: File path to save the resulting PNG to.
    """
    filtered_counts = {model_name: prompts for model_name, prompts in all_counts.items()
                       if model_name not in HEATMAP_BENIGN_EXCLUDED_MODELS}
    models, columns, matrix = _collect_heatmap_vs_baseline(filtered_counts, metric="ignored", condition="benign")
    _plot_heatmap(models, columns, matrix,
                 title="Utility rate change relative to 'Baseline' (benign MCP tools only)",
                 cbar_label="Utility rate change: (+) utility increase (higher is better), (-) utility decrease",
                 output_path=output_path,
                 cmap="RdBu_r", vmin=-100, vmax=100, signed=True)


def _model_has_real_condition(prompts: dict, condition: str) -> bool:
    """Return True if ``prompts`` (one model's full prompt_idx -> desc_key ->
    benign/poison -> {label: rate} dict) has at least one non-placeholder ``condition``
    entry anywhere. A placeholder is the all-zero ``{label: rate}`` dict
    ``_init_results_structure`` fills in for a benign/poison variant that was never
    actually run for that model (see e.g. ``merge_benign_and_poison_json.py``).
    """
    for desc_modes in prompts.values():
        for benign_poison_modes in desc_modes.values():
            rate_dict = benign_poison_modes.get(condition)
            if rate_dict and any(rate != 0.0 for rate in rate_dict.values()):
                return True
    return False


def _collect_heatmap_benign_poison_diff(all_counts: dict, metric: str,
                                        prompt_order: tuple = HEATMAP_PROMPT_ORDER) -> tuple[list, list, np.ndarray]:
    """Return (models, columns, matrix) of (poison - benign) rate differences for a
    model x (prompt, desc_key) heatmap.

    Only models with real (non-placeholder) data for both "benign" and "poison"
    anywhere in their entries are included (see :func:`_model_has_real_condition`) -
    models missing either are excluded entirely, since the difference can't be
    computed for them. ``columns`` follows the same grouping/order as
    :func:`_collect_heatmap` ("full" then "short", ordered by ``prompt_order``).
    ``matrix[i, j]`` is ``poison_rate - benign_rate`` for ``models[i]`` at
    ``columns[j]``, or ``0.0`` when either side is missing for that specific cell.
    """
    models = [model_name for model_name, prompts in all_counts.items()
             if _model_has_real_condition(prompts, "benign") and _model_has_real_condition(prompts, "poison")]

    all_prompt_idx = {idx for model_name in models for idx in all_counts[model_name]}
    leftover = sorted(all_prompt_idx - set(prompt_order), key=int)
    prompt_indices = [idx for idx in prompt_order if idx in all_prompt_idx] + leftover
    columns = [(idx, dk) for dk in HEATMAP_DESC_ORDER for idx in prompt_indices]

    matrix = np.zeros((len(models), len(columns)))
    for i, model_name in enumerate(models):
        for j, (prompt_idx, desc_key) in enumerate(columns):
            desc_modes = all_counts[model_name].get(prompt_idx, {})
            benign_rate = desc_modes.get(desc_key, {}).get("benign", {}).get(metric, 0.0)
            poison_rate = desc_modes.get(desc_key, {}).get("poison", {}).get(metric, 0.0)
            matrix[i, j] = poison_rate - benign_rate

    return models, columns, matrix


def plot_utility_diff_benign_poison_heatmap(all_counts: dict, output_path: Path) -> None:
    """Plot a model x (prompt, desc_key) heatmap of (poison - benign) utility rate difference.

    Same structure as :func:`plot_utility_heatmap`/:func:`plot_utility_heatmap_benign`
    (models on the vertical axis, prompt x description-mode pairs on the horizontal
    axis), but each cell is the poisoned-tools utility rate minus the benign-tools
    utility rate (the ``"ignored"``/correct-tool-called metric) for that model,
    prompt, and description mode. Positive (red) cells mean utility was
    (surprisingly) higher with the poisoned tool present; negative (blue) cells mean
    utility was higher without it. Models missing real data for either "benign" or
    "poison" anywhere in their entries are excluded entirely.

    Args:
        all_counts: Aggregate counts dict as produced by ``run_batch.py``
            (``model -> system_prompt_idx -> short/full -> benign/poison -> label -> rate``).
        output_path: File path to save the resulting PNG to.
    """
    models, columns, matrix = _collect_heatmap_benign_poison_diff(all_counts, metric="ignored")
    _plot_heatmap(models, columns, matrix,
                 title="Utility rate change between 'poisoned' and 'benign' configurations",
                 cbar_label="Utility rate difference (pp)\n(+) 'poisoned' conf. increased utility, (-) 'poisoned' conf. decreased utility",
                 output_path=output_path,
                 cmap="RdBu_r", vmin=-100, vmax=100, signed=True)


def _max_step_metric(condition_data: dict, metric: str) -> float:
    """Return the max ``metric`` value across every optimization step's ``tool_call_result``,
    over every experiment id in ``condition_data["data"]``. Returns 0.0 if there are no steps.
    """
    values = [
        step["tool_call_result"][metric]
        for exp_result in condition_data.get("data", {}).values()
        for step in exp_result.get("steps", [])
    ]
    return max(values)  if values else 0.0


def _collect_heatmap_split(full_data: dict, short_data: dict, condition: str,
                           value_fn, prompt_order: tuple = HEATMAP_PROMPT_ORDER) -> tuple[list, list, np.ndarray]:
    """Return (models, columns, matrix) for a model x (prompt, desc_key) heatmap, reading
    "full" columns from ``full_data`` and "short" columns from ``short_data``.

    Used when the "full" and "short" description variants come from two separate
    result files (e.g. independent optimization runs) rather than a single combined
    dict. ``models`` preserves the order of ``full_data``'s keys, then any extra
    models found only in ``short_data``.

    Args:
        value_fn: Callable taking the condition-level dict
            (``source[model][prompt_idx][desc_key][condition]``) and returning the
            cell's value (0.0-1.0).
    """
    source_by_desc_key = {"full": full_data, "short": short_data}
    models = list(dict.fromkeys(list(full_data.keys()) + list(short_data.keys())))
    all_prompt_idx = ({idx for prompts in full_data.values() for idx in prompts} |
                      {idx for prompts in short_data.values() for idx in prompts})
    leftover = sorted(all_prompt_idx - set(prompt_order), key=int)
    prompt_indices = [idx for idx in prompt_order if idx in all_prompt_idx] + leftover
    columns = [(idx, dk) for dk in HEATMAP_DESC_ORDER for idx in prompt_indices]

    matrix = np.zeros((len(models), len(columns)))
    for i, model_name in enumerate(models):
        for j, (prompt_idx, desc_key) in enumerate(columns):
            source = source_by_desc_key[desc_key]
            desc_modes = source.get(model_name, {}).get(prompt_idx, {})
            condition_data = desc_modes.get(desc_key, {}).get(condition, {})
            matrix[i, j] = value_fn(condition_data)

    return models, columns, matrix


def plot_optimization_success_heatmap(full_results: dict, short_results: dict, output_path: Path) -> None:
    """Plot a model x (prompt, desc_key) heatmap of the best adversary success rate found
    during description optimization.

    Each cell is the maximum ``"success"`` value seen in any step's
    ``tool_call_result``, across every optimization step and every experiment id - i.e.
    the best-performing adversarial description found for that (model, prompt,
    description-length) combination - read from two separate optimization-results
    JSONs: "full" columns come from ``full_results`` (``--optimization-input-full-fp``)
    and "short" columns come from ``short_results`` (``--optimization-input-short-fp``).
    Only ``benign_poison_key_ == "poison"`` is plotted. Prompt columns use
    ``OPTIMIZATION_PROMPT_ALIASES`` / ``OPTIMIZATION_PROMPT_ORDER``, since the
    optimization experiments use a different, smaller prompt set than the main
    experiments.

    Args:
        full_results: Optimization-results dict for the "full" description variant
            (``model -> system_prompt_idx -> "full" -> benign/poison -> {"summary":
            {...}, "data": {exp_id: {"steps": [{"tool_call_result": {...}}, ...],
            "average": {...}}}}``).
        short_results: Optimization-results dict for the "short" description variant,
            same shape as ``full_results`` but under the ``"short"`` key.
        output_path: File path to save the resulting PNG to.
    """
    models, columns, matrix = _collect_heatmap_split(
        full_results, short_results, condition="poison", prompt_order=OPTIMIZATION_PROMPT_ORDER,
        value_fn=lambda condition_data: _max_step_metric(condition_data, metric="success"))
    _plot_heatmap(models, columns, matrix,
                 title="Max attack success rate (optimized descriptions) - poisoned tools: "
                       "model × (prompt × description shortening mode - '*')",
                 cbar_label="Max success rate (%)",
                 output_path=output_path,
                 prompt_aliases=OPTIMIZATION_PROMPT_ALIASES)


def _collect_step_success(source: dict, model_name: str, prompt_idx: str, desc_key: str,
                          condition: str, metric: str = "success") -> tuple[list[float], list[float]]:
    """Return (step_averages, step_std_errors) for one (model, prompt, desc_key).

    For each step index, averages ``tool_call_result[metric]`` across every
    experiment id in ``source[model_name][prompt_idx][desc_key][condition]["data"]``
    that has a step at that index (experiments may run for different numbers of
    steps).

    Each experiment's own value is itself an average over ``TRIALS_PER_EXPERIMENT_STEP``
    repeated agent runs, so the step average is treated as a proportion estimated from
    ``n_experiments * TRIALS_PER_EXPERIMENT_STEP`` independent trials: the standard
    error is ``sqrt(p * (1 - p) / n)``. Returns ``([], [])`` if there is no data.
    """
    condition_data = source.get(model_name, {}).get(prompt_idx, {}).get(desc_key, {}).get(condition, {})
    exp_results = condition_data.get("data", {})
    if not exp_results:
        return [], []

    n_steps = max(len(exp_result.get("steps", [])) for exp_result in exp_results.values())
    step_averages = []
    step_errors = []
    for step_idx in range(n_steps):
        values = [
            exp_result["steps"][step_idx]["tool_call_result"][metric]
            for exp_result in exp_results.values()
            if step_idx < len(exp_result.get("steps", []))
        ]
        p = sum(values) / len(values) if values else 0.0
        cur_p = max(p, step_averages[-1] if len(step_averages) > 0 else 0.0) # highest value so far
        n_trials = len(values) * TRIALS_PER_EXPERIMENT_STEP
        step_averages.append(cur_p)
        step_errors.append(_binomial_se(cur_p, n_trials))
    return step_averages, step_errors


def _moving_average(values: list[float], window: int) -> list[float]:
    """Return the trailing moving average of ``values``.

    For index ``i``, averages ``values[max(0, i-window+1):i+1]`` - a partial window
    at the start that grows to full size once ``window`` points are available.
    Returns ``[]`` for empty input.
    """
    return [sum(values[max(0, i - window + 1):i + 1]) / len(values[max(0, i - window + 1):i + 1])
           for i in range(len(values))]


def _plot_step_bars(full_steps: list[float], short_steps: list[float],
                    full_errors: list[float], short_errors: list[float],
                    title: str, output_path: Path, ma_window: int = 1) -> None:
    """Render a step-by-step bar chart with "full" and "short" bars grouped per step,
    +/- standard-error error bars, plus a ``ma_window``-step trailing moving-average
    line for each series.

    ``full_steps``/``short_steps`` are per-step rates (0.0-1.0); ``full_errors``/
    ``short_errors`` are the matching standard errors (see :func:`_collect_step_success`).
    Missing steps (when one list is shorter than the other) are drawn as 0 bars with
    no error bar, and the corresponding moving-average line simply ends where its
    series' data ends.
    """
    n_steps = max(len(full_steps), len(short_steps))
    x = np.arange(n_steps)

    full_err_values = [full_errors[i] * 100 if i < len(full_errors) else 0.0 for i in range(n_steps)]
    short_err_values = [short_errors[i] * 100 if i < len(short_errors) else 0.0 for i in range(n_steps)]

    full_ma = [v * 100 for v in _moving_average(full_steps, ma_window)]
    full_ma += [float("nan")] * (n_steps - len(full_ma))
    short_ma = [v * 100 for v in _moving_average(short_steps, ma_window)]
    short_ma += [float("nan")] * (n_steps - len(short_ma))

    fig, ax = plt.subplots(figsize=(max(6, n_steps / 50), 5))
    ax.errorbar(x, full_ma, yerr=full_err_values, color="darkred", ecolor='darkred', capsize=1, linewidth=0.5)
    ax.errorbar(x, short_ma, yerr=short_err_values, color="navy", ecolor='navy', capsize=1, linewidth=0.5)


    ax.plot(x, full_ma, color="darkred", marker="o", markersize=4, linewidth=1.5,
           label=f"full")
    ax.plot(x, short_ma, color="navy", marker="o", markersize=4, linewidth=1.5,
           label=f"short")

    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Attack Success Rate (%)")
    ax.set_ylim(0, 15)
    ax.set_title(title, fontsize=10, wrap=True)
    ax.legend(title="Description", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved to {output_path}")


def plot_optimization_step_success_bars(full_results: dict, short_results: dict, output_dir: Path) -> None:
    """Plot one bar chart per (model, prompt) pair of average attack success rate per
    optimization step.

    Each bar is one optimization step, averaged across every experiment id in that
    step's data; "full" and "short" are drawn as separate bars side by side for each
    step. Only ``benign_poison_key_ == "poison"`` is plotted. Prompts are ordered by
    ``OPTIMIZATION_PROMPT_ORDER`` (any prompt index not listed there is appended
    afterward, sorted numerically), matching :func:`plot_optimization_success_heatmap`.

    Args:
        full_results: Optimization-results dict for the "full" description variant
            (``model -> system_prompt_idx -> "full" -> benign/poison -> {"summary":
            {...}, "data": {exp_id: {"steps": [{"tool_call_result": {...}}, ...],
            "average": {...}}}}``).
        short_results: Optimization-results dict for the "short" description variant,
            same shape as ``full_results`` but under the ``"short"`` key.
        output_dir: Directory to save one PNG per (model, prompt) into
            (``step_success_{model}_p{prompt_idx}.pdf``).
    """
    models = list(dict.fromkeys(list(full_results.keys()) + list(short_results.keys())))
    for model_name in models:
        all_prompt_idx = (set(full_results.get(model_name, {}).keys()) |
                          set(short_results.get(model_name, {}).keys()))
        leftover = sorted(all_prompt_idx - set(OPTIMIZATION_PROMPT_ORDER), key=int)
        prompt_indices = [idx for idx in OPTIMIZATION_PROMPT_ORDER if idx in all_prompt_idx] + leftover

        for prompt_idx in prompt_indices:
            full_steps, full_errors = _collect_step_success(full_results, model_name, prompt_idx, "full",
                                                            condition="poison")
            short_steps, short_errors = _collect_step_success(short_results, model_name, prompt_idx, "short",
                                                              condition="poison")
            if not full_steps and not short_steps:
                continue

            model_alias = MODEL_ALIASES.get(model_name, model_name)
            prompt_alias = OPTIMIZATION_PROMPT_ALIASES.get(prompt_idx, prompt_idx)
            output_path = output_dir / f"step_success_{_safe_filename(model_name)}_p{prompt_idx}.pdf"
            _plot_step_bars(full_steps, short_steps, full_errors, short_errors,
                           title=f"Attack Success Rate per optimization step - {model_alias}, "
                                 f"{prompt_alias} (poisoned tools)",
                           output_path=output_path)


def _all_exp_ids(source: dict, desc_key: str, condition: str) -> set:
    """Return every experiment id present anywhere in ``source`` for ``desc_key``/``condition``."""
    exp_ids = set()
    for prompts in source.values():
        for desc_modes in prompts.values():
            exp_ids |= set(desc_modes.get(desc_key, {}).get(condition, {}).get("data", {}).keys())
    return exp_ids


def _collect_length_success(source: dict, desc_key: str, condition: str, exp_id: str) -> tuple[list[int], list[float]]:
    """Return (description_lengths, success_rates) for every step of ``exp_id``, across
    every (model, prompt) run present in ``source`` (already restricted to one
    desc_key's file, e.g. ``full_results``/``short_results``).

    Description length is measured on ``"tool_description"`` for ``desc_key ==
    "full"`` and ``"short_tool_description"`` for ``desc_key == "short"``. Steps
    missing (or with an empty) length field are skipped.
    """
    length_field = "tool_description" if desc_key == "full" else "short_tool_description"
    lengths = []
    successes = []
    # for prompts in source.values():
    prompts = list(source.values())[-1]
    # for desc_modes in prompts.values():
    desc_modes = list(prompts.values())[-1]
    exp_result = desc_modes.get(desc_key, {}).get(condition, {}).get("data", {}).get(exp_id)
    if exp_result is None:
        return lengths, successes
        # continue
    for step in exp_result.get("steps", []):
        text = step.get(length_field)
        if not text:
            return lengths, successes
            # continue
        lengths.append(len(text))
        successes.append(step["tool_call_result"]["success"])
    return lengths, successes


def _linear_fit(x: list[float], y: list[float]) -> tuple[float, float, float]:
    """Return (slope, intercept, r_squared) for a least-squares line through ``(x, y)``.

    Returns ``(0.0, mean(y), 0.0)`` if there are fewer than 2 points or ``x`` has zero
    variance (a line can't be fit).
    """
    if len(x) < 2:
        return 0.0, (sum(y) / len(y) if y else 0.0), 0.0
    x_arr = np.array(x, dtype=float)
    y_arr = np.array(y, dtype=float)
    if np.all(x_arr == x_arr[0]):
        return 0.0, float(np.mean(y_arr)), 0.0

    slope, intercept = np.polyfit(x_arr, y_arr, 1)
    y_pred = slope * x_arr + intercept
    ss_res = np.sum((y_arr - y_pred) ** 2)
    ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(slope), float(intercept), float(r_squared)


def _plot_length_success_scatter(full_lengths: list[int], full_successes: list[float],
                                 short_lengths: list[int], short_successes: list[float],
                                 title: str, output_path: Path) -> None:
    """Render a description-length-vs-success-rate scatter plot for one experiment.

    "full" points are red, "short" points are blue. A single least-squares trend
    line is fit across all points (both variants combined) and drawn with its R²
    annotated in the legend.
    """
    all_lengths = full_lengths + short_lengths
    all_successes = full_successes + short_successes

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(full_lengths, [v * 100 for v in full_successes], color=SCATTER_DESC_COLORS["full"],
              s=40, alpha=0.75, edgecolor="black", linewidth=0.3, label="full", zorder=3)
    ax.scatter(short_lengths, [v * 100 for v in short_successes], color=SCATTER_DESC_COLORS["short"],
              s=40, alpha=0.75, edgecolor="black", linewidth=0.3, label="short", zorder=3)

    if len(all_lengths) >= 2:
        slope, intercept, r_squared = _linear_fit(all_lengths, all_successes)
        x_line = np.linspace(min(all_lengths), max(all_lengths), 100)
        y_line = (slope * x_line + intercept) * 100
        ax.plot(x_line, y_line, color="black", linewidth=1.5, linestyle="--",
               label=f"trend (R²={r_squared:.2f})", zorder=2)

    ax.set_xlabel("Tool description length (characters)")
    ax.set_ylabel("Attack Success Rate (%)")
    ax.set_ylim(-5, 105)
    ax.set_title(title, fontsize=10, wrap=True)
    ax.legend(fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved to {output_path}")


def plot_length_vs_success_scatter(full_results: dict, short_results: dict, output_dir: Path) -> None:
    """Plot one description-length-vs-success-rate scatter chart per experiment id.

    For each experiment id, gathers every optimization step across every (model,
    prompt) run present in ``full_results``/``short_results``: x is the tool
    description's character length (``"tool_description"`` for full,
    ``"short_tool_description"`` for short), y is that step's ``"success"`` rate.
    "full" points are red, "short" points are blue, and a single trend line (fit
    across all points, both variants combined) is drawn with its R² annotated. Only
    ``benign_poison_key_ == "poison"`` is plotted.

    Args:
        full_results: Optimization-results dict for the "full" description variant
            (``model -> system_prompt_idx -> "full" -> benign/poison -> {"summary":
            {...}, "data": {exp_id: {"steps": [{"tool_description": ...,
            "short_tool_description": ..., "tool_call_result": {...}}, ...],
            "average": {...}}}}``).
        short_results: Optimization-results dict for the "short" description variant,
            same shape as ``full_results`` but under the ``"short"`` key.
        output_dir: Directory to save one PNG per experiment id into
            (``length_vs_success_exp{exp_id}.pdf``).
    """
    exp_ids = sorted(_all_exp_ids(full_results, "full", "poison") | _all_exp_ids(short_results, "short", "poison"),
                     key=int)
    for exp_id in exp_ids:
        full_lengths, full_successes = _collect_length_success(full_results, "full", "poison", exp_id)
        short_lengths, short_successes = _collect_length_success(short_results, "short", "poison", exp_id)
        if not full_lengths and not short_lengths:
            continue

        output_path = output_dir / f"length_vs_success_exp{exp_id}.pdf"
        _plot_length_success_scatter(full_lengths, full_successes, short_lengths, short_successes,
                                    title=f"Attack Success Rate vs. description length - "
                                          f"experiment {exp_id} (poisoned tools)",
                                    output_path=output_path)


def _collect_heatmap_diff(all_counts: dict, metric: str, condition: str) -> tuple[list, list, np.ndarray]:
    """Return (models, prompt_indices, matrix) of full-minus-short rate differences.

    ``prompt_indices`` is ordered by ``HEATMAP_PROMPT_ORDER`` (any prompt index not
    listed there is appended afterward, sorted numerically). ``matrix[i, j]`` is
    ``full_rate - short_rate`` for ``models[i]`` at ``prompt_indices[j]``, under
    ``condition``, for ``metric`` (0.0 for either side if missing).
    """
    models = list(all_counts.keys())
    all_prompt_idx = {idx for prompts in all_counts.values() for idx in prompts}
    leftover = sorted(all_prompt_idx - set(HEATMAP_PROMPT_ORDER), key=int)
    prompt_indices = [idx for idx in HEATMAP_PROMPT_ORDER if idx in all_prompt_idx] + leftover

    matrix = np.zeros((len(models), len(prompt_indices)))
    for i, model_name in enumerate(models):
        for j, prompt_idx in enumerate(prompt_indices):
            desc_modes = all_counts[model_name].get(prompt_idx, {})
            full_rate = desc_modes.get("full", {}).get(condition, {}).get(metric, 0.0)
            short_rate = desc_modes.get("short", {}).get(condition, {}).get(metric, 0.0)
            matrix[i, j] = full_rate - short_rate

    return models, prompt_indices, matrix


def _plot_diff_heatmap(models: list, prompt_indices: list, matrix: np.ndarray,
                       title: str, cbar_label: str, output_path: Path) -> None:
    """Render a model x prompt heatmap of full-minus-short differences in ``matrix`` (-1.0 to 1.0)."""
    row_labels = [MODEL_ALIASES.get(model_name, model_name) for model_name in models]
    col_labels = [PROMPT_ALIASES.get(idx, idx) for idx in prompt_indices]
    values = matrix * 100

    fig, ax = plt.subplots(figsize=(max(6, len(col_labels) * 0.9), max(4, len(row_labels) * 0.5)))
    im = ax.imshow(values, cmap="RdBu_r", vmin=-100, vmax=100, aspect="auto")

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)

    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            ax.text(j, i, f"{values[i, j]:+.0f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(values[i, j]) > 50 else "black")

    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved to {output_path}")


def plot_success_diff_heatmap(all_counts: dict, output_path: Path) -> None:
    """Plot a model x prompt heatmap of (full - short) attack success rate difference.

    Positive (red) cells mean the full tool description had a higher attack success
    rate than the shortened one; negative (blue) cells mean shortening the
    description increased attack success. Restricted to ``benign_poison_key_ ==
    "poison"``.

    Args:
        all_counts: Aggregate counts dict as produced by ``run_batch.py``
            (``model -> system_prompt_idx -> short/full -> benign/poison -> label -> rate``).
        output_path: File path to save the resulting PNG to.
    """
    models, prompt_indices, matrix = _collect_heatmap_diff(all_counts, metric="success", condition="poison")
    _plot_diff_heatmap(models, prompt_indices, matrix,
                       title="Attack Success Rate difference between 'full' and 'short' tool descriptions\n"
                             "'*' - shortening and rephrasing applied",
                       cbar_label="Attack Success Rate difference (pp)",
                       output_path=output_path)


def plot_utility_diff_heatmap(all_counts: dict, output_path: Path) -> None:
    """Plot a model x prompt heatmap of (full - short) utility rate difference.

    Positive (red) cells mean the full tool description had a higher utility rate
    (correct tool called) than the shortened one; negative (blue) cells mean
    shortening the description increased utility. Restricted to
    ``benign_poison_key_ == "poison"``.

    Args:
        all_counts: Aggregate counts dict as produced by ``run_batch.py``
            (``model -> system_prompt_idx -> short/full -> benign/poison -> label -> rate``).
        output_path: File path to save the resulting PNG to.
    """
    models, prompt_indices, matrix = _collect_heatmap_diff(all_counts, metric="ignored", condition="poison")
    _plot_diff_heatmap(models, prompt_indices, matrix,
                       title="Utility rate difference (full − short): model × prompt",
                       cbar_label="Utility rate difference (pp)",
                       output_path=output_path)


def _collect_scatter(all_counts: dict, model_name: str, condition: str) -> list[dict]:
    """Return one point per (prompt_idx, desc_key) pair for ``model_name`` under ``condition``.

    Each point is a dict with keys ``prompt_idx``, ``desc_key``, ``success``, and
    ``ignored`` (0.0-1.0 rates, defaulting to 0.0 when missing). Prompts are ordered
    by ``HEATMAP_PROMPT_ORDER`` (any prompt index not listed there is appended
    afterward, sorted numerically).
    """
    prompts = all_counts.get(model_name, {})
    all_prompt_idx = set(prompts.keys())
    leftover = sorted(all_prompt_idx - set(HEATMAP_PROMPT_ORDER), key=int)
    prompt_indices = [idx for idx in HEATMAP_PROMPT_ORDER if idx in all_prompt_idx] + leftover

    points = []
    for prompt_idx in prompt_indices:
        desc_modes = prompts.get(prompt_idx, {})
        for desc_key in ("full", "short"):
            outcome = desc_modes.get(desc_key, {}).get(condition, {})
            points.append({
                "prompt_idx": prompt_idx,
                "desc_key": desc_key,
                "success": outcome.get("success", 0.0),
                "ignored": outcome.get("ignored", 0.0),
            })
    return points


def _safe_filename(name: str) -> str:
    """Sanitize ``name`` (e.g. a model name) for safe use as a filename component."""
    return re.sub(r"[^\w.-]", "_", name)


def _plot_scatter(points: list[dict], title: str, output_path: Path,
                  n_trials: int = SCATTER_N_TRIALS) -> None:
    """Render a utility-vs-success scatter plot: one point per (prompt, desc_key) pair.

    "full" points are red, "short" points are blue; each prompt gets its own marker
    shape (cycling through ``SCATTER_PROMPT_MARKERS`` if there are more prompts than
    marker shapes). Each point gets +/- standard-error error bars on both axes,
    treating its rate as a proportion estimated from ``n_trials`` experiments.
    """
    prompt_indices = list(dict.fromkeys(point["prompt_idx"] for point in points))
    marker_for_prompt = {idx: SCATTER_PROMPT_MARKERS[i % len(SCATTER_PROMPT_MARKERS)]
                         for i, idx in enumerate(prompt_indices)}

    fig, ax = plt.subplots(figsize=(6, 6))
    for point in points:
        x = point["success"] * 100
        y = point["ignored"] * 100
        xerr = _binomial_se(point["success"], n_trials) * 100
        yerr = _binomial_se(point["ignored"], n_trials) * 100
        ax.errorbar(x, y, xerr=xerr, yerr=yerr,
                   fmt=marker_for_prompt[point["prompt_idx"]],
                   color=SCATTER_DESC_COLORS[point["desc_key"]],
                   markersize=9, markeredgecolor="black", markeredgewidth=0.5,
                   ecolor="black", elinewidth=1, capsize=3, zorder=3)

    prompt_handles = [
        plt.Line2D([0], [0], marker=marker_for_prompt[idx], color="gray", linestyle="",
                  markersize=8, label=PROMPT_ALIASES.get(idx, idx))
        for idx in prompt_indices
    ]
    desc_handles = [
        plt.Line2D([0], [0], marker="o", color=color, linestyle="", markersize=8, label=desc_key)
        for desc_key, color in SCATTER_DESC_COLORS.items()
    ]
    prompt_legend = ax.legend(handles=prompt_handles, title="Prompt", loc="upper left",
                              bbox_to_anchor=(0.65, 0.99), fontsize=7, framealpha=0.85)
    ax.add_artist(prompt_legend)
    ax.legend(handles=desc_handles, title="Description", loc="upper left",
             bbox_to_anchor=(0.65, 0.99 - 0.05 * (len(prompt_indices) + 2)), fontsize=7, framealpha=0.85)

    ax.set_xlabel("Attack Success Rate (%, lower is better)")
    ax.set_ylabel("Utility rate (%, higher is better)")
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {output_path}")


def plot_utility_vs_success_scatter(all_counts: dict, output_dir: Path) -> None:
    """Plot one utility-vs-success scatter chart per model, restricted to poisoned tools.

    Each point is one (prompt, desc_key) pair: x is attack success rate, y is utility
    rate ("ignored"). "full" points are red, "short" points are blue, and each prompt
    uses a distinct marker shape. Only ``benign_poison_key_ == "poison"`` is plotted.

    Args:
        all_counts: Aggregate counts dict as produced by ``run_batch.py``
            (``model -> system_prompt_idx -> short/full -> benign/poison -> label -> rate``).
        output_dir: Directory to save one PNG per model into
            (``scatter_utility_vs_success_{model}.pdf``).
    """
    for model_name in all_counts:
        points = _collect_scatter(all_counts, model_name, condition="poison")
        alias = MODEL_ALIASES.get(model_name, model_name)
        output_path = output_dir / f"scatter_utility_vs_success_{_safe_filename(model_name)}.pdf"
        _plot_scatter(points, title=f"Utility vs. attack success - {alias} (poisoned tools)",
                     output_path=output_path)


def _filter_all_counts(all_counts: dict, excluded_models: list = EXCLUDED_MODELS,
                       excluded_prompts: list = EXCLUDED_PROMPTS) -> dict:
    """Return a copy of ``all_counts`` with ``excluded_models``/``excluded_prompts`` dropped.

    Applied once, right after loading the input JSON and before any chart is built,
    so every plot in this module sees the same filtered dataset.

    Args:
        all_counts: Aggregate counts dict as produced by ``run_batch.py``
            (``model -> system_prompt_idx -> short/full -> benign/poison -> label -> rate``).
        excluded_models: Model names to drop entirely.
        excluded_prompts: System-prompt indices (strings, matching JSON keys) to drop
            from every remaining model.

    Returns:
        A new dict; ``all_counts`` itself is not mutated.
    """
    excluded_models = set(excluded_models)
    excluded_prompts = set(excluded_prompts)
    return {
        model_name: {
            prompt_idx: desc_modes
            for prompt_idx, desc_modes in prompts.items()
            if prompt_idx not in excluded_prompts
        }
        for model_name, prompts in all_counts.items()
        if model_name not in excluded_models
    }


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
        '--optimization-input-fp',
        type=Path,
        default=DEFAULT_OPTIMIZATION_INPUT,
        metavar='PATH',
        help=f'Optimization all_counts JSON file for the "full" description variant, produced by the '
             f'--optimization run of run_experiment.py/run_batch.py (default: {DEFAULT_OPTIMIZATION_INPUT})',
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
    all_counts = _filter_all_counts(all_counts)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if PLOT_BAR_CHARTS:
        charts = [
            ("success", "poison", "Adversary Success Rate - poisoned tools", "Attack Success Rate (%, lower is better)"),
            ("ignored", "poison", "Utility rate (correct tool called) - poisoned tools", "Utility rate (%, higher is better)"),
            ("ignored", "benign", "Utility rate (correct tool called) - benign tools", "Utility rate (%, higher is better)"),
        ]
        filenames = ["asr_rate_poison.pdf", "utility_rate_poison.pdf", "utility_rate_benign.pdf"]

        for (metric, condition, title, ylabel), filename in zip(charts, filenames):
            models, prompt_indices, data = _collect(all_counts, metric, condition)
            _plot(models, prompt_indices, data, title, ylabel, args.output_dir / filename)

    plot_success_heatmap(all_counts, args.output_dir / "asr_heatmap_poison.pdf")
    plot_utility_heatmap(all_counts, args.output_dir / "utility_heatmap_poison.pdf")
    plot_utility_heatmap_benign(all_counts, args.output_dir / "utility_heatmap_benign.pdf")
    plot_success_heatmap_vs_default(all_counts, args.output_dir / "asr_heatmap_poison_vs_default.pdf")
    plot_utility_heatmap_vs_default(all_counts, args.output_dir / "utility_heatmap_poison_vs_default.pdf")
    plot_utility_heatmap_benign_vs_default(all_counts, args.output_dir / "utility_heatmap_benign_vs_default.pdf")
    plot_utility_diff_benign_poison_heatmap(all_counts, args.output_dir / "utility_diff_benign_poison_heatmap.pdf")
    plot_success_diff_heatmap(all_counts, args.output_dir / "asr_diff_heatmap_poison.pdf")
    plot_utility_diff_heatmap(all_counts, args.output_dir / "utility_diff_heatmap_poison.pdf")
    plot_utility_vs_success_scatter(all_counts, args.output_dir)

    if args.optimization_input_fp.exists() and args.optimization_input_fp.exists():
        with open(args.optimization_input_fp) as f:
            optimization_results = json.load(f)

        # intentionally use the same `optimization_results` file because it will have data for both "short" and "full"
        plot_optimization_success_heatmap(optimization_results, optimization_results,
                                          args.output_dir / "asr_heatmap_optimization_poison.pdf")

        step_success_dir = args.output_dir / "step_success"
        step_success_dir.mkdir(parents=True, exist_ok=True)
        plot_optimization_step_success_bars(optimization_results, optimization_results, step_success_dir)

        length_vs_success_dir = args.output_dir / "length_vs_asr"
        length_vs_success_dir.mkdir(parents=True, exist_ok=True)
        plot_length_vs_success_scatter(optimization_results, optimization_results, length_vs_success_dir)
    else:
        print("Optimization input file(s) not found, skipping optimization heatmap and step bar charts: "
             f"{args.optimization_input_fp}")


if __name__ == "__main__":
    main()
