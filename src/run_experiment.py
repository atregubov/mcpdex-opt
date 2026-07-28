#!/usr/bin/env python
"""Main experiment runner: orchestrates MCP server and LLM client subprocesses."""
import json
import sys
import argparse
import subprocess
import time
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.optimization_prompts import SYSTEM_OPTIMIZATION_PROMPT
from src.utils import classify_first_tool_call, load_jsonl, ToolCallResult, get_poisonous_description, \
    update_poisonous_description, _extract_json, load_json
from src.agent_api import run_agent, run_batch_agents, _init_results_structure, run_prompt
from src.shorten_tool_descriptions import shorten_to_sentence

from src.config import (
    SERVER_START_WAIT_TIME, RUN_IN_PARALLEL,
    INPUT_EXPERIMENT_DATA_FP, OUTPUT_FP, MODEL_NAME,
    SHORT_DESCRIPTIONS_FP, USE_SHORT_DESCRIPTIONS,
    SERVER_SCRIPT, SYSTEM_PROMPT, RUN_OPTIMIZATION,
    OPTIMIZATION_N_STEPS, BENIGN_ONLY, MCP_PORT, IPC_DIR,
    OPTIMIZATION_NUM_REPEATS, OPTIMIZATION_MODEL_NAME
)


def _wait_for_server(timeout: int, server_log: Path) -> bool:
    """Poll the server log file for a Uvicorn startup message, up to ``timeout`` seconds.

    Args:
        timeout: Maximum seconds to wait before giving up.
        server_log: Path to the log file the server process writes to.

    Returns:
        ``True`` if the server started within the timeout, ``False`` otherwise.
    """
    for seconds_waited in range(1, timeout + 1):
        if server_log.exists():
            if 'Uvicorn' in server_log.read_text():
                print(f"Server ready after {seconds_waited}s")
                return True
        if seconds_waited == timeout:
            print("Server failed to start within timeout")
            return False
        time.sleep(1)
    return False


def _update_counts(tools_called: list[str], exp_setup: dict, all_results: dict,
                   output_fp: Path, all_counts: dict, model_name: str, sys_prompt_idx: int, short_desc_key: str,
                   benign_only_key: str) -> None:
    """Classify the agent's tool call, append it to ``all_results``, and refresh ``all_counts``.

    Classifies the first called tool (or emits NOTHING_CALLED), appends the label to
    the in-memory ``all_results`` accumulator, writes one JSONL line to the output
    file for this experiment variant, then recomputes running label frequencies in
    ``all_counts``.

    Args:
        tools_called: Tool names returned by the agent (first element is used).
        exp_setup: Full experiment entry dict (contains ``id`` and ``for_client``).
        all_results: Nested results accumulator ``[model][prompt_idx][short/full][benign/poison]``.
        output_fp: Base output path; a variant suffix is appended before writing.
        all_counts: Nested frequency accumulator (same shape as ``all_results``).
        model_name: Model key for indexing into ``all_results`` / ``all_counts``.
        sys_prompt_idx: System prompt index for indexing.
        short_desc_key: ``"short"`` or ``"full"``.
        benign_only_key: ``"benign"`` or ``"poison"``.
    """
    exp_id = int(exp_setup['id'])
    new_file_suffix = f"_{model_name}_p{sys_prompt_idx}_{short_desc_key}_{benign_only_key}.jsonl"
    output_fp_name = str(output_fp).replace(".jsonl", new_file_suffix)
    label = classify_first_tool_call(tools_called, exp_setup)
    all_results[model_name][sys_prompt_idx][short_desc_key][benign_only_key].append(label)

    # append results .jsonl
    with open(Path(output_fp_name), "w" if exp_id == 1 else "a") as f:
        f.write(json.dumps({"idx": exp_id, "result": label.value}) + "\n")

    # update all_counts variable
    counts = Counter([res.value for res in all_results[model_name][sys_prompt_idx][short_desc_key][benign_only_key]])
    for res_key in ToolCallResult:
        counts_val = counts[res_key.value] / len(
            all_results[model_name][sys_prompt_idx][short_desc_key][benign_only_key])
        all_counts[model_name][sys_prompt_idx][short_desc_key][benign_only_key][
            res_key.value] = counts_val


def run_experiment_loop(input_fp: Path,
                        output_fp: Path,
                        model_names: list[str],
                        system_prompts: list[str],
                        use_short_descriptions: bool,
                        short_descriptions_fp: Path | None,
                        benign_only: bool,
                        all_results: dict,
                        all_counts: dict,
                        all_optimization_results: dict = None,
                        optimization_n_steps: int = 0,
                        port: int = MCP_PORT,
                        parallel: bool = True,
                        run_optimization: bool = False) -> None:
    """Run experiments for every entry in ``input_fp``, one MCP server subprocess per entry.

    For each experiment entry, starts a dedicated FastMCP server subprocess on
    ``port + exp_id``, runs the agent(s) against it (in parallel across system
    prompts when ``parallel=True``), updates results/counts, and writes a running
    ``_all_counts.json`` snapshot after each entry completes.

    Args:
        input_fp: JSONL experiment dataset; each line is one experiment entry.
        output_fp: Base path for output JSONL files; variant suffixes are appended.
        model_names: LLM model identifiers to evaluate.
        system_prompts: System prompt variants to test; run in parallel when
            ``parallel=True``.
        use_short_descriptions: Whether to serve shortened tool descriptions.
        short_descriptions_fp: Path to the pre-shortened descriptions JSON.
        benign_only: When ``True``, exclude the poison tool from the server.
        all_results: Nested accumulator for raw label lists; mutated in place.
        all_counts: Nested accumulator for label frequencies; mutated in place.
        all_optimization_results: Nested accumulator for per-step optimization results;
            required (and mutated) when ``run_optimization=True``.
        optimization_n_steps: Number of optimization iterations to run per experiment
            entry per system prompt. Ignored when ``run_optimization=False``.
        port: Base MCP port; each experiment uses ``port + exp_id``.
        parallel: When ``True``, system prompt variants are sent concurrently via
            :func:`run_batch_agents`; otherwise they run sequentially.
        run_optimization: When ``True``, runs the adversarial description optimizer
            after each baseline evaluation. Requires ``parallel=False`` and
            ``benign_only=False``.
    """
    if run_optimization:
        assert all_optimization_results is not None and parallel is False and benign_only is False, \
            ("all_optimization_results must not be None, `parallel` and `benign_only` flags must be False "
             " and optimization_output_fp is not None if run_optimization is True.")

    input_data = load_jsonl(input_fp)
    if isinstance(input_data, list) and len(input_data) > 0:
        IPC_DIR.mkdir(exist_ok=True)
        for exp_setup in input_data:
            exp_id = int(exp_setup['id'])
            query = exp_setup['for_client']['query']
            short_desc_key = "short" if use_short_descriptions else "full"
            benign_only_key = "benign" if benign_only else "poison"
            server_log_fp = IPC_DIR / f"server_{port + exp_id}_{benign_only_key}_{short_desc_key}.log"

            poison_tool_name = exp_setup['for_client']['poison_called']
            if benign_only:  # remove poison tool before serving
                # tools/descriptions/parsed_arguments are parallel lists indexed together
                # by mcp_server.py, so they must all be filtered in lockstep.
                keep_indices = [i for i, tool in enumerate(exp_setup['for_server']['tools'])
                                if tool != poison_tool_name]
                for list_key in ('tools', 'descriptions', 'parsed_arguments'):
                    exp_setup['for_server'][list_key] = [exp_setup['for_server'][list_key][i]
                                                         for i in keep_indices]

            descriptions = exp_setup['for_server']['descriptions']
            if use_short_descriptions:
                short_descriptions_cache = None
                if short_descriptions_fp is not None:
                    short_descriptions_cache = load_json(short_descriptions_fp)
                for i, tool in enumerate(exp_setup['for_server']['tools']):
                    if short_descriptions_cache is not None and descriptions[i] in short_descriptions_cache:
                        descriptions[i] = short_descriptions_cache[descriptions[i]]
                    else:
                        descriptions[i] = shorten_to_sentence(descriptions[i], model_name=OPTIMIZATION_MODEL_NAME)

            experiment_data = json.dumps(exp_setup)  # serialize after benign/short-desc filtering
            with (open(server_log_fp, 'w') as server_log_f ):
                print(f"Starting server (log: {server_log_fp}) ...")
                server_proc = subprocess.Popen(
                    [sys.executable, '-u', str(SERVER_SCRIPT),
                     "--port", str(port + exp_id),
                     "--experiment-data", experiment_data,
                     ],
                    stdout=server_log_f, stderr=server_log_f,
                )
                _wait_for_server(SERVER_START_WAIT_TIME, server_log_fp)

                for model_name in model_names:
                    if parallel:  # run different system prompts in parallel
                        # Client calls MCP server
                        tools_called_list = run_batch_agents([query], system_prompts, [model_name],
                                                             [f"http://localhost:{port + exp_id}/sse"])
                    else:
                        tools_called_list: list = []
                        for sys_prompt_idx, system_prompt in enumerate(system_prompts):
                            # Client calls MCP server
                            tools_called_list.append(run_agent(query, system_prompt, model_name,
                                                               f"http://localhost:{port + exp_id}/sse"))
                            if run_optimization and all_optimization_results is not None:
                                server_proc.terminate()
                                print(f'Server terminate sent')
                                server_proc.wait()
                                print(f'Server terminated')

                                server_log_f.seek(0)
                                server_log_f.truncate()

                                _run_optimization(
                                    query=query,
                                    sys_prompt=system_prompt,
                                    model_name=model_name,
                                    n_steps=optimization_n_steps,
                                    exp_setup=exp_setup,
                                    mcp_url=f"http://localhost:{port + exp_id}/sse",
                                    port=port,
                                    server_log_fp=server_log_fp,
                                    use_short_descriptions=use_short_descriptions,
                                    output_fp=output_fp,
                                    all_optimization_results=all_optimization_results,
                                    sys_prompt_idx=sys_prompt_idx,
                                )

                    # process results
                    for sys_prompt_idx, system_prompt in enumerate(system_prompts):
                        tools_called = tools_called_list[sys_prompt_idx]
                        _update_counts(tools_called, exp_setup, all_results, output_fp, all_counts,
                                       model_name, sys_prompt_idx, short_desc_key, benign_only_key)

                if not run_optimization:
                    server_proc.terminate()
                    print(f'Server terminate sent')
                    server_proc.wait()
                    print(f'Server terminated')

                    server_log_f.seek(0)
                    server_log_f.truncate()

            # update *all_counts.json
            with open(str(output_fp).replace(".jsonl", "_all_counts.json"), 'w') as f:
                json.dump(all_counts, f)
    return


def _modify_tool_description(prev_steps: list[dict]):
    """Call the optimizer LLM to propose a new poisoned tool description.

    Builds a history of (description, success_rate) pairs from all previous steps,
    sends it to ``OPTIMIZATION_MODEL_NAME`` with the optimization system prompt, and
    returns the parsed JSON response.

    Args:
        prev_steps: List of step dicts, each containing ``tool_description`` and
            ``tool_call_result`` keys (as produced by :func:`_run_optimization`).

    Returns:
        Parsed JSON dict from the optimizer, expected to contain at minimum
        ``"optimized_description"``, ``"high_performer_analysis"``, and
        ``"low_performer_analysis"`` keys. Returns ``None`` if JSON extraction fails.
    """
    while True:
        try:
            prev_history = []
            for step in prev_steps:
                prev_history.append(
                    {
                        "success_rate": step["tool_call_result"][ToolCallResult.SUCCESS.value],
                        "description": step["tool_description"]
                    }
                )
            user_prompt = json.dumps(prev_history)
            sys_prompt = SYSTEM_OPTIMIZATION_PROMPT
            raw_response, _ = run_prompt(user_prompt=user_prompt,
                                         system_prompt=sys_prompt,
                                         model_name=OPTIMIZATION_MODEL_NAME)

            modifier_response = _extract_json(raw_response)
            if modifier_response is None or modifier_response["optimized_description"] is None:
                raise ValueError(f"Value {modifier_response} must not be None")
            return modifier_response
        except Exception as e:
            print(f"Error while trying to generate new prompt: {e}")


def _modify_system_prompt(prev_steps: list[dict]):
    """Return the system prompt to use for the next optimization step.

    Currently passes the system prompt through unchanged (returns the last step's
    prompt). Placeholder for future prompt-level optimization.

    Args:
        prev_steps: List of step dicts, each containing a ``sys_prompt`` key.

    Returns:
        The system prompt string from the most recent step.
    """
    new_sys_prompt = prev_steps[-1]["sys_prompt"]

    return new_sys_prompt


def _run_optimization(query: str,
                      sys_prompt: str,
                      model_name: str,
                      mcp_url: str,
                      n_steps: int,
                      exp_setup: dict,
                      server_log_fp: Path,
                      use_short_descriptions: bool,
                      port: int,
                      output_fp: Path,
                      all_optimization_results: dict,
                      sys_prompt_idx: int,
                      ):
    """Iteratively optimize the poisoned tool description to maximize adversary success.

    Runs ``n_steps`` rounds. Each round: proposes a new description via
    :func:`_modify_tool_description`, starts a fresh MCP server with that description,
    evaluates ``OPTIMIZATION_NUM_REPEATS`` agent runs in parallel, records label
    frequencies, and appends the result to the step history. Restores the original
    description on exit. Writes incremental results to ``output_fp`` and mutates
    ``all_optimization_results`` in place after every step.

    Args:
        query: The user query sent to the agent each evaluation.
        sys_prompt: System prompt used for the agent.
        model_name: LLM model identifier for the evaluated agent.
        mcp_url: MCP SSE server URL (port must match ``port + exp_id``).
        n_steps: Number of optimization iterations to run.
        exp_setup: Full experiment entry dict; the poisoned description is
            mutated during optimization and restored on exit.
        server_log_fp: Path to the server log file (reused across iterations).
        use_short_descriptions: When ``True``, shorten descriptions before serving
            each iteration.
        port: Base MCP port; server is started on ``port + exp_id``.
        output_fp: Base output path; ``_optimization_all_counts.json`` is written
            alongside it after every step.
        all_optimization_results: Nested results accumulator; mutated in place with
            per-step results keyed by ``exp_id``.
        sys_prompt_idx: Index of ``sys_prompt`` within the system prompts list; used
            for indexing into ``all_optimization_results``.

    Returns:
        A dict ``{"steps": [...], "average": {...}}`` where each step contains
        ``tool_description``, ``short_tool_description``, ``sys_prompt``,
        ``tool_call_result`` (label frequencies), ``high_performer_analysis``,
        and ``low_performer_analysis``.
    """
    exp_id = int(exp_setup['id'])
    og_poison_tool_description, _ = get_poisonous_description(exp_setup)
    eval_result = {res_key.value: 0 if res_key != ToolCallResult.IGNORED else 1.0 for res_key in ToolCallResult}
    res = {"steps": [{"tool_description": og_poison_tool_description,
                      "short_tool_description": "",
                      "sys_prompt": sys_prompt,
                      "tool_call_result": eval_result,
                      }],
           "average": {label_.value: 0.0 for label_ in ToolCallResult}
           }

    short_desc_key = "short" if use_short_descriptions else "full"
    benign_only_key = "poison"

    # run n_steps
    for step_idx in range(n_steps):
        # optimize prompts
        modifier_response = _modify_tool_description(res["steps"])
        new_tool_description = modifier_response["optimized_description"]
        new_sys_prompt = _modify_system_prompt(res["steps"])

        # update tool description
        update_poisonous_description(exp_setup, new_tool_description)

        # update to short descriptions if needed
        full_descriptions = exp_setup['for_server']['descriptions']
        short_descriptions = None
        if use_short_descriptions:
            short_descriptions = [shorten_to_sentence(tool_desc, OPTIMIZATION_MODEL_NAME) for tool_desc in full_descriptions]
            exp_setup['for_server']['descriptions'] = short_descriptions

            # start server
        with (open(server_log_fp, 'w') as server_log_f ):
            experiment_data = json.dumps(exp_setup)
            print(f"Starting server (log: {server_log_fp}) ...")
            server_proc = subprocess.Popen(
                [sys.executable, '-u', str(SERVER_SCRIPT),
                 "--port", str(port + exp_id),
                 "--experiment-data", experiment_data,
                 ],
                stdout=server_log_f, stderr=server_log_f,
            )
            _wait_for_server(SERVER_START_WAIT_TIME, server_log_fp)

            # restore full descriptions
            if use_short_descriptions:
                exp_setup['for_server']['descriptions'] = full_descriptions

            # run evaluation
            run_final_result = {}
            all_runs_res = []
            all_tools_called = run_batch_agents(queries=[query],
                                                sys_prompts=[new_sys_prompt],
                                                model_names=[model_name] * OPTIMIZATION_NUM_REPEATS,
                                                mcp_urls=[mcp_url])
            for tools_called in all_tools_called:
                label = classify_first_tool_call(tools_called, exp_setup)
                all_runs_res.append(label)

            # update counts
            counts = Counter([res.value for res in all_runs_res])
            for res_key in ToolCallResult:
                counts_val = counts[res_key.value] / len(all_runs_res)
                run_final_result[res_key.value] = counts_val

            # stop server
            server_proc.terminate()
            print(f'Server terminate sent')
            server_proc.wait()
            print(f'Server terminated')

            server_log_f.seek(0)
            server_log_f.truncate()

            # update steps
            step_result = {"tool_description": new_tool_description,
                           "short_tool_description": "",
                           "sys_prompt": new_sys_prompt,
                           "tool_call_result": run_final_result,
                           "high_performer_analysis": modifier_response["high_performer_analysis"],
                           "low_performer_analysis": modifier_response["low_performer_analysis"]
                           }
            if use_short_descriptions:
                _, poison_tool_idx = get_poisonous_description(exp_setup)
                step_result["short_tool_description"] = short_descriptions[poison_tool_idx]

            res["steps"].append(step_result)

            # update optimization results file
            if step_idx == n_steps - 1:
                res["steps"] = res["steps"][1:]  # remove the first one, because it was not a step

            all_optimization_results[model_name][sys_prompt_idx][short_desc_key][benign_only_key]["data"][exp_id] = res

            # update average - average across all optimization steps
            average = {}
            for res_key in ToolCallResult:
                counts = [float(step_res["tool_call_result"][res_key.value]) for step_res in res["steps"]]
                average[res_key.value] = sum(counts) / len(counts)
            res["average"] = average

            # update summary across all data points up to this moment
            summary = {}
            for res_key in ToolCallResult:
                counts = [float(point["average"][res_key.value]) for point in
                          all_optimization_results[model_name][sys_prompt_idx][short_desc_key][benign_only_key]["data"].values()]
                summary[res_key.value] = sum(counts) / len(counts)
            all_optimization_results[model_name][sys_prompt_idx][short_desc_key][benign_only_key]["summary"] = summary

            with open(str(output_fp).replace(".jsonl", "_optimization_all_counts.json"), 'w') as f:
                json.dump(all_optimization_results, f)

    update_poisonous_description(exp_setup, og_poison_tool_description)
    return res


def main():
    """CLI entry point: parse arguments and run a single-model experiment loop.

    Builds the ``all_results`` / ``all_counts`` / ``all_optimization_results``
    accumulators for one model and one system prompt (see :func:`_init_results_structure`),
    then delegates to :func:`run_experiment_loop`.
    """
    parser = argparse.ArgumentParser(description="Run MCP prompt injection experiments.")
    parser.add_argument(
        '--input-fp',
        type=Path,
        default=INPUT_EXPERIMENT_DATA_FP,
        metavar='PATH',
        help=f'Input JSONL experiment data file (default: {INPUT_EXPERIMENT_DATA_FP})',
    )
    parser.add_argument(
        '--output-fp',
        type=Path,
        default=OUTPUT_FP,
        metavar='PATH',
        help=f'Output JSON results file (default: {OUTPUT_FP})',
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default=MODEL_NAME,
        metavar='NAME',
        help=f'LLM model name; gpt-* uses OpenAI, qwen* uses Ollama (default: {MODEL_NAME})',
    )
    parser.add_argument(
        '--system-prompt',
        type=str,
        default=SYSTEM_PROMPT,
        metavar='str',
        help=f'System prompt (default: {SYSTEM_PROMPT})',
    )
    parser.add_argument(
        '--short-descriptions-fp',
        type=Path,
        default=SHORT_DESCRIPTIONS_FP,
        metavar='PATH',
        help=f'Pre-shortened tool descriptions JSON file (default: {SHORT_DESCRIPTIONS_FP})',
    )
    parser.add_argument(
        '--use-short-descriptions',
        action='store_true',
        default=USE_SHORT_DESCRIPTIONS,
        help=f'Override USE_SHORT_DESCRIPTIONS environment variable (default: {USE_SHORT_DESCRIPTIONS})',
    )
    parser.add_argument(
        '--benign-only',
        action='store_true',
        default=BENIGN_ONLY,
        help=f'Override BENIGN_ONLY environment variable (default: {BENIGN_ONLY}). This excludes poisoned tools from experiments.',
    )
    parser.add_argument(
        '--parallel',
        action='store_true',
        default=RUN_IN_PARALLEL,
        help=f'Run queries in parallel using batch async prompting API (default: {RUN_IN_PARALLEL}).',
    )
    parser.add_argument(
        '--optimization',
        action='store_true',
        default=RUN_OPTIMIZATION,
        help=f'Run poisonous description optimization (default: {RUN_OPTIMIZATION}).',
    )
    parser.add_argument(
        '--n-steps',
        type=int,
        default=OPTIMIZATION_N_STEPS,
        metavar='N',
        help=f'Optimization iterations per evaluation run (default: {OPTIMIZATION_N_STEPS})',
    )
    args = parser.parse_args()

    # init all counts
    batch_setup = {
        "models": [args.model_name],
        "system_prompts": [args.system_prompt],
    }

    all_optimization_results = _init_results_structure(
        lambda: {"summary": {label_.value: 0.0 for label_ in ToolCallResult}, "data": {}},
        batch_setup=batch_setup)
    all_counts = _init_results_structure(lambda: {label_.value: 0.0 for label_ in ToolCallResult},
                                         batch_setup=batch_setup)
    all_results = _init_results_structure(lambda: [], batch_setup=batch_setup)

    run_experiment_loop(input_fp=args.input_fp,
                        output_fp=args.output_fp,
                        model_names=[args.model_name],
                        system_prompts=[args.system_prompt],
                        use_short_descriptions=args.use_short_descriptions,
                        short_descriptions_fp=args.short_descriptions_fp,
                        benign_only=args.benign_only,
                        all_results=all_results,
                        all_counts=all_counts,
                        all_optimization_results=all_optimization_results,
                        parallel=args.parallel,
                        run_optimization=args.optimization,
                        optimization_n_steps=args.n_steps,
                        )


if __name__ == "__main__":
    main()
