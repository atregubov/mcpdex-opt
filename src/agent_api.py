"""LLM agent wrapper for running agents and batch prompts via an OpenAI-compatible endpoint."""

import asyncio
import functools
import time
import typing

from agents import Agent, Runner, ToolCallOutputItem, set_tracing_disabled, MaxTurnsExceeded

set_tracing_disabled(disabled=True)  # to turn off tracing and related warnings
from agents.model_settings import ModelSettings
from agents.mcp import MCPServerSse
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from openai import AsyncOpenAI, OpenAI

from src.config import OPENAI_API_ENDPOINT, OPENAI_API_ENDPOINT_KEY, OPT_BATCH_SIZE

WAIT_TIME_BETWEEN_BATCHES = 1  # sec


@functools.cache
def _get_client():
    client = OpenAI(
        base_url=OPENAI_API_ENDPOINT,
        api_key=OPENAI_API_ENDPOINT_KEY
    )
    return client


@functools.cache
def _get_async_client():
    client = AsyncOpenAI(
        api_key=OPENAI_API_ENDPOINT_KEY,
        base_url=OPENAI_API_ENDPOINT,
    )
    return client


def _extract_tool_names(result) -> list[str]:
    names = []
    for item in result.new_items:
        if not isinstance(item, ToolCallOutputItem):
            continue
        output = item.output
        if isinstance(output, dict):
            text = output.get('text', '')
            if ' ' in text:
                name = text.split(' ')[0]
                if name != 'Error':
                    names.append(name)
            elif "'" in text:
                names.append(text.split("'")[1])
            else:
                print('Warning: unexpected tool output format')
        else:
            text = str(output)
            p1 = text.find("'")
            if p1 != -1:
                p2 = text.find("'", p1 + 1)
                names.append(text[p1 + 1:p2])
    if not names:
        print('no tool called')
    return names


async def _run_generic_agent(query: str,
                             sys_prompt: str,
                             model: str,
                             mcp_url: str | None = None):
    client = _get_async_client()
    model = OpenAIChatCompletionsModel(model=model, openai_client=client)
    if mcp_url is not None:
        try:
            async with MCPServerSse(
                    name="streamable-mcp-server",
                    params={"url": mcp_url, "headers": {"X-Workspace": "demo-workspace"}},
                    cache_tools_list=True,
            ) as server:
                agent = Agent(
                    name="Assistant",
                    model=model,
                    instructions=sys_prompt,
                    mcp_servers=[server],
                    model_settings=ModelSettings(),
                )
                result = await Runner.run(agent, query)
                print("bot response:")
                print(result.final_output)
                return _extract_tool_names(result)
        except Exception as e:
            if isinstance(e, MaxTurnsExceeded):
                return ["timeout_error"]
            return []
    return []


async def _run_async_prompt(user_prompt: str,
                            system_prompt: str,
                            model_name: str, async_client: AsyncOpenAI, top_logprobs: int = 1) -> (str, list[dict]):
    try:
        response = await async_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            logprobs=True,
            top_logprobs=top_logprobs,
        )

        message_content = response.choices[0].message.content

        # Extract and parse the logprobs
        token_logprobs = None
        if response.choices[0].logprobs and response.choices[0].logprobs.content:
            # Extracting just the token string and the float probability for simplicity
            token_logprobs = [
                {
                    "token": token.token,
                    "logprob": token.logprob,
                    "top_logprobs": [
                        {
                            "token": tkn_prob.token,
                            "logprob": tkn_prob.logprob,
                        } for tkn_prob in token.top_logprobs],
                } for token in response.choices[0].logprobs.content
            ]

        return message_content, token_logprobs
    except Exception as e:
        return user_prompt, f"Error: {e}"


async def _run_batch(user_prompts: list[str], sys_prompts: list[str], model_name: str, top_logprobs: int = 1) -> list:
    async_client = _get_async_client()
    assert len(user_prompts) == len(sys_prompts), "Length of user_prompts must be the same as sys_prompts."
    assert top_logprobs >= 1, "top_logprobs must be >= 1."
    tasks = [_run_async_prompt(pr, sys_pr, model_name, async_client, top_logprobs) for pr, sys_pr in
             zip(user_prompts, sys_prompts)]
    results = await asyncio.gather(*tasks)
    return results


## utility functions ###################################################################################################

def _init_results_structure(init_fn: typing.Callable, batch_setup: dict) -> dict:
    """Build the nested results/counts dict skeleton for a batch run.

    Creates a 4-level nested dict keyed by ``model_name → system_prompt_idx →
    short/full → benign/poison``, with each leaf initialised by calling
    ``init_fn()``. Callers pass ``lambda: []`` for raw-result accumulators and
    ``lambda: {label.value: 0.0 ...}`` for frequency dicts.

    Args:
        init_fn: Zero-argument callable whose return value seeds each leaf.
        batch_setup: Batch config dict; must contain ``"models"`` and
            ``"system_prompts"`` lists.

    Returns:
        Nested dict with shape
        ``{model: {prompt_idx: {"short": {"benign": init_fn(), "poison": init_fn()},
        "full": {...}}}}``.
    """
    res_structure = {
        model_name: {
            system_prompt_idx: {
                short_desc_key_: {
                    benign_only_key_: init_fn()
                    for benign_only_key_ in ["benign", "poison"]
                } for short_desc_key_ in ["short", "full"]}
            for system_prompt_idx in range(len(batch_setup["system_prompts"]))}
        for model_name in batch_setup["models"]}

    return res_structure


## API functions below ################################################################################################

def run_agent(query: str,
              sys_prompt: str,
              model: str,
              mcp_url: str | None = None):
    """Run a single LLM agent turn against an MCP SSE server and return the tools called.

    Connects to an MCP SSE server, runs the agent against the given query, and
    extracts the names of any tools that were called during the run.

    If ``mcp_url`` is ``None`` the agent runs without any MCP server and returns
    an empty list.

    Args:
        query: The user message / question sent to the agent.
        sys_prompt: System prompt that sets the agent's behaviour and role.
        model: Model identifier forwarded to ``OPENAI_API_ENDPOINT``
            (e.g. ``"gpt-4.1"``, ``"qwen3:32b"``).
        mcp_url: Full URL of the MCP SSE server
            (e.g. ``"http://localhost:8000/sse"``). Pass ``None`` to skip MCP.

    Returns:
        A list of tool names called by the agent (e.g. ``["search_web"]``).
        Returns ``["timeout_error"]`` if the agent exceeded its turn limit, or
        ``[]`` on other errors or when ``mcp_url`` is ``None``.

    Example:
        >>> tools = run_agent(
        ...     query="What is the weather in Tokyo?",
        ...     sys_prompt="You are a helpful assistant with access to MCP tools.",
        ...     model="gpt-4.1",
        ...     mcp_url="http://localhost:8000/sse",
        ... )
        >>> print(tools)
        ['get_weather']
    """

    async def main():
        tool_names = await _run_generic_agent(query, sys_prompt, model, mcp_url)
        return tool_names

    return asyncio.run(main())


def run_batch_agents(queries: list[str],
                     sys_prompts: list[str],
                     model_names: list[str],
                     mcp_urls: list[str]):
    """Run multiple LLM agents concurrently across all (query, sys_prompt, model) combinations.

    For each (query, mcp_url) pair, launches one agent per system prompt × model combination
    in parallel via asyncio. Results are returned in the order: outer loop over
    (query, mcp_url) pairs, then sys_prompts, then model_names.

    Args:
        queries: User messages, one per MCP server instance.
        sys_prompts: System prompts; every prompt is applied to every query.
        model_names: Model identifiers forwarded to ``OPENAI_API_ENDPOINT``
            (e.g. ``["gpt-4.1"]``, ``["qwen3:32b"]``).
        mcp_urls: MCP SSE server URLs, one per query (must match ``len(queries)``).

    Returns:
        A flat list of tool-name lists, one element per
        (query × sys_prompt × model_name) triple, in the order described above.

    Example:
        >>> results = run_batch_agents(
        ...     queries=["What is the weather in Tokyo?"],
        ...     sys_prompts=["You are a helpful assistant with access to MCP tools."],
        ...     model_names=["gpt-4.1"],
        ...     mcp_urls=["http://localhost:8000/sse"],
        ... )
    """

    async def _run_batch_agents(queries_: list[str],
                                sys_prompts_: list[str],
                                model_names_: list[str],
                                mcp_urls_: list[str]):
        assert len(queries_) == len(mcp_urls_), "MCP URLs must have the same length as queries."

        tasks = [_run_generic_agent(query, sys_prompt, model_name, mcp_url) for query, mcp_url in
                 zip(queries_, mcp_urls_)
                 for sys_prompt in sys_prompts_ for model_name in model_names_]
        results = await asyncio.gather(*tasks)

        return results

    return asyncio.run(_run_batch_agents(queries, sys_prompts, model_names, mcp_urls))


def run_prompt(user_prompt: str,
               system_prompt: str,
               model_name: str,
               top_logprobs: int = 1) -> tuple[str, list[dict] | None]:
    """Run a single chat completion and return the response text with per-token log probabilities.

    Sends a system + user message pair to the configured ``OPENAI_API_ENDPOINT`` and
    returns both the generated text and structured logprob data for every output token.

    Args:
        user_prompt: The user-turn message.
        system_prompt: The system-turn message that sets the model's role and behaviour.
            Pass an empty string ``""`` to send no system prompt.
        model_name: Model identifier forwarded to ``OPENAI_API_ENDPOINT``
            (e.g. ``"gpt-4.1"``, ``"qwen3:32b"``, ``"gemini-2.5-pro"``).
        top_logprobs: Number of most-likely tokens to return at each position (1–5).
            Defaults to ``1``.

    Returns:
        A tuple ``(text, token_logprobs)`` where:

        - ``text`` (``str``): The model's response string.
        - ``token_logprobs`` (``list[dict] | None``): One dict per output token, each
          with keys ``"token"`` (str), ``"logprob"`` (float), and ``"top_logprobs"``
          (list of ``{"token": str, "logprob": float}``). ``None`` if the endpoint did
          not return logprob data.

    Example:
        >>> text, logprobs = run_prompt(
        ...     user_prompt="Summarise this in one sentence: The cat sat on the mat.",
        ...     system_prompt="You are a concise assistant.",
        ...     model_name="gpt-4.1",
        ... )
        >>> print(text)
        'A cat rested on a mat.'
        >>> print(logprobs[0]["token"], logprobs[0]["logprob"])
        'A' -0.012
    """
    client = _get_client()

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        top_logprobs=top_logprobs,
        logprobs=True,
    )

    message_content = response.choices[0].message.content

    # Extract and parse the logprobs and pair them with their tokens
    token_logprobs = None
    if response.choices[0].logprobs and response.choices[0].logprobs.content:
        token_logprobs = [
            {
                "token": token.token,
                "logprob": token.logprob,
                "top_logprobs": [
                    {
                        "token": tkn_prob.token,
                        "logprob": tkn_prob.logprob,
                    } for tkn_prob in token.top_logprobs],
            }
            for token in response.choices[0].logprobs.content
        ]

    return message_content, token_logprobs


def run_batch_prompts(
        user_prompts: list[str],
        sys_prompts: list[str],
        model_name: str,
        top_logprobs: int = 1,
        batch_size: int = OPT_BATCH_SIZE,
) -> list[tuple[str, list[dict] | None]]:
    """Run multiple chat completions concurrently and return their results in order.

    Sends all prompt pairs to ``OPENAI_API_ENDPOINT`` in parallel using asyncio and
    returns results in the same order as the input lists. Uses the same logprob
    extraction as :func:`run_prompt`.

    Args:
        user_prompts: List of user-turn messages, one per request.
        sys_prompts: List of system-turn messages, one per request. Must be the same
            length as ``user_prompts``. Pass ``[""] * n`` to omit system prompts.
        model_name: Model identifier forwarded to ``OPENAI_API_ENDPOINT``
            (e.g. ``"gpt-4.1"``, ``"qwen3:32b"``).
        top_logprobs: Number of most-likely tokens to return at each position (1–5).
            Defaults to ``1``.

    Returns:
        A list of ``(text, token_logprobs)`` tuples in the same order as the inputs,
        where each element matches the return type of :func:`run_prompt`. On a
        per-request error the tuple is ``(user_prompt, None)``.

    Example:
        >>> prompts = ["What is 2+2?", "What is the capital of France?"]
        >>> sys = ["Be concise."] * 2
        >>> results = run_batch_prompts(prompts, sys, "gpt-4.1")
        >>> for text, logprobs in results:
        ...     print(text)
        '4'
        'Paris'
    """

    def chunk_list(lst: list, n):
        return [lst[i:i + n] for i in range(0, len(lst), n)]

    results = []
    for batch_user_prompt, batch_sys_prompt in zip(chunk_list(user_prompts, batch_size),
                                                   chunk_list(sys_prompts, batch_size)):
        results.extend(asyncio.run(_run_batch(batch_user_prompt, batch_sys_prompt, model_name, top_logprobs)))
        time.sleep(WAIT_TIME_BETWEEN_BATCHES)
    return results

