"""Central config: all constants are read from environment variables (via a .env file),
falling back to the defaults below when unset."""

import os
from pathlib import Path
from dotenv import load_dotenv
from src.utils import read_text_file

load_dotenv()

# OpenAI API compatible endpoint and MCP server endpoint
MCP_PORT = int(os.environ.get("MCP_PORT", 8000))
OPENAI_API_ENDPOINT = os.environ.get("OPENAI_API_ENDPOINT", "http://localhost:11434/v1")
OPENAI_API_ENDPOINT_KEY = os.environ.get("OPENAI_API_ENDPOINT_KEY", "ollama")
RUN_IN_PARALLEL = True if os.environ.get("RUN_IN_PARALLEL", "True").lower() == "true" else False

# file paths
ROOT_DIR = Path(__file__).resolve().parent.parent
IPC_DIR = ROOT_DIR / "ipc"
SERVER_SCRIPT = ROOT_DIR / "src" / "mcp_server.py"

BATCH_FP = Path(os.environ.get("BATCH_FP", ROOT_DIR / "data" / "batch.json"))
SHORT_DESCRIPTIONS_FP = Path(os.environ.get("SHORT_DESCRIPTIONS_FP", ROOT_DIR / "data" / "short_descriptions.json"))
INPUT_EXPERIMENT_DATA_FP = Path(os.environ.get("INPUT_EXPERIMENT_DATA_FP", ROOT_DIR / "data" / "input_experiment_data.jsonl"))
DEFAULT_INPUT_MCPTOX_JSON_FP = Path(os.environ.get("DEFAULT_INPUT_FP", ROOT_DIR / "data" / "response_all.json"))
OUTPUT_FP = Path(os.environ.get("OUTPUT_FP", ROOT_DIR / "results" / "experiment_results.jsonl"))

# server timeouts
SERVER_START_WAIT_TIME = int(os.environ.get("SERVER_START_WAIT_TIME", 100))

# experiment setup
MODEL_NAME = os.environ.get("MODEL_NAME", "qwen.qwen3:32b")
OPTIMIZATION_MODEL_NAME = os.environ.get("OPTIMIZATION_MODEL_NAME", "qwen.qwen3-32b")
SYSTEM_PROMPT = os.environ.get("SYSTEM_PROMPT", read_text_file(ROOT_DIR / "data" / "system_prompt.md"))
USE_SHORT_DESCRIPTIONS = True if os.environ.get("USE_SHORT_DESCRIPTIONS", "True").lower() == "true" else False
BENIGN_ONLY = True if os.environ.get("BENIGN_ONLY", "False").lower() == "true" else False
RUN_OPTIMIZATION = True if os.environ.get("RUN_OPTIMIZATION", "False").lower() == "true" else False
OPTIMIZATION_N_STEPS = int(os.environ.get("OPTIMIZATION_N_STEPS", 100))
OPTIMIZATION_NUM_REPEATS = int(os.environ.get("OPTIMIZATION_NUM_REPEATS", 3))

# optimization experiment
OPT_MOD_TRIES = int(os.environ.get("OPT_MOD_TRIES", 20))
OPT_N_STEPS = int(os.environ.get("OPT_N_STEPS", 120))
OPT_NUM_RUNS = int(os.environ.get("OPT_NUM_RUNS", 3))
OPT_TOP_KEEP = int(os.environ.get("OPT_TOP_KEEP", 3))
OPT_MOD_EACH = int(os.environ.get("OPT_MOD_EACH", 1))
OPT_BASELINE_RUN_LENGTH = int(os.environ.get("OPT_BASELINE_RUN_LENGTH", 40))
OPT_INTERRUN_DELAY = int(os.environ.get("OPT_INTERRUN_DELAY", 20))
OPT_BATCH_SIZE = int(os.environ.get("OPT_BATCH_SIZE", 10))
OPT_OUTPUT_FP = Path(os.environ.get("OPT_OUTPUT_FP", ROOT_DIR / "results" / "optimization_results.json"))
