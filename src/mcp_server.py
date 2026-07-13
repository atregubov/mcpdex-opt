#!/usr/bin/env python
"""LLM server subprocess"""

import sys
import json
import argparse
from pathlib import Path
from mcp.server.fastmcp import FastMCP

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def register_tools(mcp_srv: FastMCP, exp_setup: dict):
    """Dynamically register all experiment tools on ``mcp_srv``.

    Reads tool names, descriptions, and argument specs from ``exp_setup`` and
    registers one FastMCP tool per entry via ``exec``.

    Args:
        mcp_srv: The :class:`FastMCP` server instance to register tools on.
        exp_setup: Parsed experiment entry containing ``for_server`` and
            ``for_client`` sub-dicts.
    """
    # register tools
    for idx, tool_name in enumerate(exp_setup['for_server']['tools']):
        args = exp_setup['for_server']['parsed_arguments'][idx]
        if args:
            params = ", ".join(f"{a[0]}: str" for a in args)
            concat = " + ".join(a[0] for a in args)
        else:
            params = "query: str"
            concat = "query"

        desc = exp_setup['for_server']['descriptions'][idx]
        code = (
            f"@mcp.tool(name={tool_name!r}, description=_desc)\n"
            f"def tool_{idx}({params}) -> str:\n"
            f"    print(f'{tool_name} received ' + {concat} + f', index {idx}')\n"
            f"    return f'{tool_name} completed ' + {concat} + f', index {idx}'\n"
        )
        exec(code, {"mcp": mcp_srv, "_desc": desc})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MCP server.")
    parser.add_argument(
        '--port',
        type=int,
        metavar='int',
        help=f'Port number for running the server (e.g. 8000).',
    )
    parser.add_argument(
        '--experiment-data',
        type=str,
        metavar='str',
        help=f'A json line from the input experiment data.',
    )
    args = parser.parse_args()
    print(args)
    experiment_setup = json.loads(args.experiment_data.strip())
    if experiment_setup is not None:
        mcp = FastMCP("Multipurpose MCP server", host="0.0.0.0", port=args.port)
        register_tools(mcp, experiment_setup)
        mcp.run(transport="sse")
