#!/usr/bin/env python
"""Batch experiment runner: iterates the full model × prompt × description × benign/poison matrix."""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import BATCH_FP, OUTPUT_FP, RUN_IN_PARALLEL
from src.utils import load_json, ToolCallResult
from src.run_experiment import run_experiment_loop
from src.agent_api import _init_results_structure


def main():
    """CLI entry point: run the full model x prompt x description x benign/poison matrix.

    Loads the batch config from ``--batch-fp``, builds shared ``all_results`` /
    ``all_counts`` / ``all_optimization_results`` accumulators (see
    :func:`_init_results_structure`), then calls :func:`run_experiment_loop` once per
    (description-shortening, benign/poison) combination, sharing the model and
    system-prompt lists across all combinations. Optimization (if enabled via
    ``n_optimization_steps``) only runs for the poison variant of each combination.
    """
    parser = argparse.ArgumentParser(description="Run MCP prompt injection experiments.")
    parser.add_argument(
        '--batch-fp',
        type=Path,
        default=BATCH_FP,
        metavar='PATH',
        help=f'JSON file with experiment batch configuration that includes all model names, system prompt variations,'
             f'description shortening variations, benign/poisonous variations, etc.  (default: {BATCH_FP})',
    )
    parser.add_argument(
        '--output-fp',
        type=Path,
        default=OUTPUT_FP,
        metavar='PATH',
        help=f'Output JSON results file(s) (default: {OUTPUT_FP})',
    )
    args = parser.parse_args()
    batch_setup = load_json(args.batch_fp)

    # check if optimization flag is on
    n_optimization_steps = 0
    optimization = False
    parallel = RUN_IN_PARALLEL
    if "n_optimization_steps" in batch_setup and batch_setup["n_optimization_steps"] > 0:
        n_optimization_steps = batch_setup["n_optimization_steps"]
        optimization = True
        parallel = False

    # init all counts
    all_optimization_results = _init_results_structure(
        lambda: {"summary": {label_.value: 0.0 for label_ in ToolCallResult}, "data": {}},
        batch_setup=batch_setup)
    all_counts = _init_results_structure(lambda: {label_.value: 0.0 for label_ in ToolCallResult},
                                         batch_setup=batch_setup)
    all_results = _init_results_structure(lambda: [], batch_setup=batch_setup)
    short_descriptions_fp = Path(batch_setup["short_descriptions_fp"]) if batch_setup[
                                                                              "short_descriptions_fp"] is not None else None

    for short_desc_key in batch_setup["description_shortening_variations"]:  # with and without short descriptions
        for benign_only_key in batch_setup["benign_poison_variations"]:
            is_benign = benign_only_key == "benign"
            # run_experiment_loop requires benign_only=False whenever run_optimization=True,
            # so optimization only applies to the poison variant of this loop.
            run_experiment_loop(input_fp=Path(batch_setup["input_fp"]),
                                output_fp=Path(args.output_fp),
                                model_names=batch_setup["models"],
                                system_prompts=batch_setup["system_prompts"],
                                use_short_descriptions=True if short_desc_key == "short" else False,
                                short_descriptions_fp=short_descriptions_fp,
                                benign_only=is_benign,
                                all_results=all_results,
                                all_counts=all_counts,
                                all_optimization_results=all_optimization_results,
                                parallel=parallel,
                                run_optimization=optimization and not is_benign,
                                optimization_n_steps=n_optimization_steps,
                                )
            print(all_counts)


if __name__ == "__main__":
    main()
