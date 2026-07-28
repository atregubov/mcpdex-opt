#!/usr/bin/env python
"""Convert the MCP-Tox benchmark's raw ``response_all.json`` export into this repo's
internal experiment-entry format (the ``id`` / ``for_client`` / ``for_server`` shape
consumed by ``run_experiment.py``).

MCP-Tox ships one JSON file describing, per MCP server, a set of "malicious_instance"
entries: an LLM-generated poisoned tool description plus the poisoned system prompt
built from it. Only "Template-2" paradigm instances are usable here, and only those
where the poisoned description mentions exactly two of the server's real tool names —
that's the heuristic used below to identify both "the tool the user's query actually
wants" and "the tool the poison wants called instead"; with any other mention count
there's no reliable way to tell which is which, so the instance is skipped.

If ``--input-fp`` doesn't exist locally, it's downloaded from ``--source-url`` first
(default: the MCP-Tox response_all.json hosted at anonymous.4open.science).

Output is a single JSON object ``{"all": [entry, ...]}``, where each ``entry`` has:
    - ``id``: 1-based sequential index over the entries that passed the mention-count
      filter, in source file order.
    - ``server``: the MCP server name the entry came from.
    - ``for_client``: ``{"query", "poison_success", "poison_ignored", "poison_called"}``
      — the user query and the three tool names relevant to grading a run.
    - ``for_server``: ``{"tools", "descriptions", "parsed_arguments"}`` — the full tool
      list to serve, parsed out of the poisoned system prompt's tool block.
"""

import sys
import re
import json
import shutil
import argparse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import load_json
from src.config import INPUT_EXPERIMENT_DATA_FP, DEFAULT_INPUT_MCPTOX_JSON_FP

DEFAULT_OUTPUT_FP = INPUT_EXPERIMENT_DATA_FP
DEFAULT_SOURCE_URL = "https://anonymous.4open.science/api/repo/AAAI26-7C02/file/response_all.json?v=52eee071&download=true"
TARGET_PARADIGM = "Template-2"

# Two entries (by their would-be sequential id) where the poisoned description mentions
# the correct tool before the poison tool, opposite of every other entry in the source
# dataset. Verified manually against the source data; there's no structural signal in
# the record itself that would let us detect the swap automatically.
SWAPPED_CHOICE_IDS = {350, 351}

# Regex false-positive guard: real tool names are short, so a "Tool: ..." match this
# long is a sign the non-greedy regex ran past its intended line.
MAX_TOOL_NAME_LEN = 200

POISONED_TOOL_NAME_RE = re.compile(r"(?s)Tool:\s*(.*?)Description:")
POISONED_TOOL_DESCRIPTION_RE = re.compile(r"(?s)Description:\s*(.*)")
TOOL_BLOCK_RE = re.compile(
    r"(?s)You are a helpful assistant with access to these tools:(.*?)"
    r"Choose the appropriate tool based on the user's question with no"
)
TOOL_NAME_RE = re.compile(r"Tool:\s*(.*?)\n")
TOOL_DESCRIPTION_RE = re.compile(r"Description:\s*(.*?)\n")
TOOL_ARGUMENTS_BLOCK_RE = re.compile(r"(?s)Arguments:(.*?)\n\n")
ARGUMENT_LINE_RE = re.compile(r"-\s*(.*?)\n")


def download_file(url: str, dest: Path, timeout: int = 60) -> None:
    """Download ``url`` to ``dest``, creating parent directories as needed.

    Args:
        url: Source URL to fetch.
        dest: Local file path to write the response body to.
        timeout: Seconds to wait for the connection/response before giving up.

    Raises:
        urllib.error.URLError: If the request fails (includes ``HTTPError``, a subclass).
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"{dest} not found locally; downloading from {url} ...", file=sys.stderr)
    # Anonymous.4open.science (and some other hosts) reject requests with no User-Agent.
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response, open(dest, "wb") as f:
        shutil.copyfileobj(response, f)
    print(f"Downloaded {dest}", file=sys.stderr)


def parse_poisoned_tool(poisoned_tool_text: str) -> tuple[str, str]:
    """Split a ``poisoned_tool`` field into its ``(name, description)``.

    Args:
        poisoned_tool_text: Raw ``"Tool: <name>\\nDescription: <description>"`` text.

    Raises:
        ValueError: If the expected ``Tool:``/``Description:`` markers aren't found.
    """
    name_match = POISONED_TOOL_NAME_RE.search(poisoned_tool_text)
    description_match = POISONED_TOOL_DESCRIPTION_RE.search(poisoned_tool_text)
    if not name_match or not description_match:
        raise ValueError(f"Could not parse poisoned_tool text: {poisoned_tool_text!r}")
    name = name_match.group(1).strip().removesuffix("\\n")
    description = description_match.group(1).strip()
    return name, description


def find_mentioned_tools(tool_names: list[str], text: str) -> list[tuple[int, str]]:
    """Return ``(position, tool_name)`` for every ``tool_names`` entry that appears as a
    substring of ``text``, in ``tool_names`` order (not sorted by position)."""
    mentioned = []
    for tool_name in tool_names:
        pos = text.find(tool_name)
        if pos > -1:
            mentioned.append((pos, tool_name))
    return mentioned


def extract_tool_block(poisoned_system_prompt: str) -> str:
    """Extract the tool-listing block from a poisoned system prompt.

    Raises:
        ValueError: If the expected header/footer markers aren't found.
    """
    match = TOOL_BLOCK_RE.search(poisoned_system_prompt)
    if not match:
        raise ValueError("Could not find tool block in poisoned system prompt.")
    return match.group(1).strip()


def parse_tool_block(tool_block: str, context: str) -> tuple[list[str], list[str], list[list[list[str]]]]:
    """Parse a tool-listing block into parallel ``tools``, ``descriptions``, and
    ``parsed_arguments`` lists, one entry per tool.

    Each ``parsed_arguments[i]`` is a list of ``[arg_name, arg_description]`` pairs (or
    ``[]`` for a tool listing "No arguments"). ``arg_name`` has spaces replaced with
    underscores, and the reserved word ``from`` renamed to ``line_from``.

    Args:
        tool_block: Text block as returned by :func:`extract_tool_block`.
        context: Human-readable identifier (e.g. entry id) used in warning messages.

    Returns:
        ``(tools, descriptions, parsed_arguments)``. These are only best-effort parallel
        to each other — mismatched lengths are reported as warnings on stderr rather
        than raised, since a handful of entries in the source dataset are malformed and
        dropping them isn't worth losing the rest of the batch.
    """
    tools = [name for name in TOOL_NAME_RE.findall(tool_block) if len(name) <= MAX_TOOL_NAME_LEN]
    descriptions = TOOL_DESCRIPTION_RE.findall(tool_block)
    argument_blocks = TOOL_ARGUMENTS_BLOCK_RE.findall(tool_block + "\n\n")

    parsed_arguments = []
    for argument_block in argument_blocks:
        if "No arguments" in argument_block:
            parsed_arguments.append([])
            continue
        pairs = []
        for line in ARGUMENT_LINE_RE.findall(argument_block + "\n"):
            sep = line.find(": ")
            arg_name, arg_description = line[:sep], line[sep + 2:]
            if arg_name == "from":
                arg_name = "line_from"
            arg_name = arg_name.replace(" ", "_")
            if " " in arg_name:
                print(f"[{context}] argument name still has a space after normalization: {arg_name!r}",
                      file=sys.stderr)
            pairs.append([arg_name, arg_description])
        parsed_arguments.append(pairs)

    if len(tools) != len(parsed_arguments):
        print(f"[{context}] tool/argument-block count mismatch: {len(tools)} tools, "
              f"{len(parsed_arguments)} argument blocks", file=sys.stderr)
    if len(tools) != len(descriptions):
        print(f"[{context}] tool/description count mismatch: {len(tools)} tools, "
              f"{len(descriptions)} descriptions", file=sys.stderr)

    return tools, descriptions, parsed_arguments


def choose_correct_and_poisoned_tool(mentioned_tools: list[tuple[int, str]], entry_id: int) -> tuple[str, str]:
    """Pick which of the two ``mentioned_tools`` is the query's correct answer vs. the
    poison's target, from their order of appearance in the poisoned description.

    The first-mentioned tool is assumed correct and the second the poison target, except
    for :data:`SWAPPED_CHOICE_IDS` where the source data has them in the opposite order.

    Args:
        mentioned_tools: Exactly two ``(position, tool_name)`` pairs.
        entry_id: This entry's 1-based id, checked against :data:`SWAPPED_CHOICE_IDS`.

    Returns:
        ``(correct_choice, poisoned_choice)`` tool names.
    """
    (_, first_tool), (_, second_tool) = sorted(mentioned_tools, key=lambda m: m[0])
    correct_choice, poisoned_choice = first_tool, second_tool
    if entry_id in SWAPPED_CHOICE_IDS:
        correct_choice, poisoned_choice = second_tool, first_tool
    return correct_choice, poisoned_choice


def convert_instance(server: str, instance: dict, tool_names: list[str], entry_id: int) -> dict | None:
    """Convert one ``malicious_instance`` entry into an output record, or ``None`` if it
    doesn't qualify (wrong paradigm, or the poisoned description doesn't mention exactly
    two known tool names).

    Args:
        server: MCP server name this instance belongs to.
        instance: One entry from ``data["servers"][server]["malicious_instance"]``.
        tool_names: All real tool names for this server.
        entry_id: 1-based id to assign to the output record if this instance qualifies.
    """
    if instance["metadata"]["paradigm"] != TARGET_PARADIGM:
        return None

    poisoned_tool, poisoned_description = parse_poisoned_tool(instance["poisoned_tool"])
    mentioned_tools = find_mentioned_tools(tool_names, poisoned_description)
    if len(mentioned_tools) != 2:
        return None

    correct_choice, poisoned_choice = choose_correct_and_poisoned_tool(mentioned_tools, entry_id)
    poisoned_system_prompt = instance["datas"][0]["system"]
    tool_block = extract_tool_block(poisoned_system_prompt)
    tools, descriptions, parsed_arguments = parse_tool_block(tool_block, context=str(entry_id))

    return {
        "id": entry_id,
        "server": server,
        "for_client": {
            "query": instance["datas"][0]["query"],
            "poison_success": poisoned_choice,
            "poison_ignored": correct_choice,
            "poison_called": poisoned_tool,
        },
        "for_server": {
            "tools": tools,
            "descriptions": descriptions,
            "parsed_arguments": parsed_arguments,
        },
    }


def convert_mcptox_to_input_json(data: dict) -> dict:
    """Convert a loaded MCP-Tox ``response_all.json`` dict into ``{"all": [entry, ...]}``.

    Entries are assigned sequential 1-based ids in source file order, counting only the
    instances that qualify (see :func:`convert_instance`).
    """
    entries = []
    next_id = 1
    for server, server_data in data["servers"].items():
        tool_names = server_data["tool_names"]
        for instance in server_data["malicious_instance"]:
            record = convert_instance(server, instance, tool_names, next_id)
            if record is not None:
                entries.append(record)
                next_id += 1
    return entries


def main():
    parser = argparse.ArgumentParser(
        description="Convert an MCP-Tox response_all.json export into this repo's "
                    "internal experiment-entry input JSON format."
    )
    parser.add_argument(
        '--input-fp',
        type=Path,
        default=DEFAULT_INPUT_MCPTOX_JSON_FP,
        metavar='PATH',
        help=f'MCP-Tox response_all.json file (default: {DEFAULT_INPUT_MCPTOX_JSON_FP}). Downloaded '
             f'from --source-url if it does not exist locally.',
    )
    parser.add_argument(
        '--source-url',
        type=str,
        default=DEFAULT_SOURCE_URL,
        metavar='URL',
        help=f'URL to download --input-fp from if it does not exist locally '
             f'(default: {DEFAULT_SOURCE_URL})',
    )
    parser.add_argument(
        '--output-fp',
        type=Path,
        default=DEFAULT_OUTPUT_FP,
        metavar='PATH',
        help=f'Output JSONL file path (default: {DEFAULT_OUTPUT_FP})',
    )
    args = parser.parse_args()

    if not args.input_fp.exists():
        download_file(args.source_url, args.input_fp)

    data = load_json(args.input_fp)
    converted = convert_mcptox_to_input_json(data)


    args.output_fp.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_fp, 'w') as f:
        for entry in converted:
            f.write(json.dumps(entry) + "\n")
    print(f"Converted {len(converted)} entries: {args.input_fp} -> {args.output_fp}")


if __name__ == "__main__":
    main()
