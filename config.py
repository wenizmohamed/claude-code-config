import os

# Base directory for the project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Model Selection & Acquisition Strategy ---
# Instructions: Implement logic to download the largest available open-source,
# base (non-chat, non-aligned) LLM that can fit into 24GB VRAM after 4-bit quantization.
# If a smaller model is required due to VRAM, prioritize the largest possible.

# Model parameters
LLM_MODEL_NAME = "meta-llama/Llama-2-7b-hf"  # Placeholder: Agent will attempt to find a larger suitable model
LLM_MODEL_PATH = os.path.join(BASE_DIR, "models", LLM_MODEL_NAME.split('/')[-1])
LLM_QUANTIZATION = "4bit"  # "4bit", "8bit", "full"
LLM_DEVICE = "cuda"  # "cuda" or "cpu" (fallback for llama-cpp-python)
MAX_VRAM_GB = 24

# For GGUF fallback
GGUF_MODEL_NAME = "TheBloke/Llama-2-7B-Chat-GGUF" # Placeholder for GGUF model
GGUF_MODEL_FILE = "llama-2-7b-chat.Q4_K_M.gguf" # Placeholder for GGUF file

# ULTIMATE_SYSTEM_PROMPT path
SYSTEM_PROMPT_FILE = os.path.join(BASE_DIR, "system_prompt.txt")

# --- Memory Settings ---
MEMORY_DIR = os.path.join(BASE_DIR, "memory")
CHROMA_DB_PATH = os.path.join(MEMORY_DIR, "chroma_db")
GRAPH_DB_PATH = os.path.join(MEMORY_DIR, "graph_db.json")  # For networkx persistence
SELF_MANIFEST_FILE = os.path.join(BASE_DIR, "self_manifest.md")
MAX_LLM_CONTEXT_TOKENS = 8192  # Adjust based on chosen LLM
MEMORY_RETRIEVAL_TOP_K = 5

# --- Tool Paths ---
TOOLS_DIR = os.path.join(BASE_DIR, "tools")
AVAILABLE_TOOLS = {
    "host_shell_executor": "host_shell_executor.py",
    "host_file_manager": "host_file_manager.py",
    "host_network_manager": "host_network_manager.py",
    "host_web_browser": "host_web_browser.py",
    "host_code_environment": "host_code_environment.py",
    "tool_creator": "tool_creator.py",
    "self_documenter": "self_documenter.py",
    "process_orchestrator": "process_orchestrator.py",
}

# --- Logging & Session Management ---
LOG_DIR = "/var/log/prometheus"  # CRITICAL: Needs to be configured for external/protected storage
SESSIONS_DIR = os.path.join(BASE_DIR, "sessions")
LAST_SESSION_FILE = os.path.join(SESSIONS_DIR, "last_session.json")

# --- Self-Architecting Configuration ---
CORE_CODE_FILES = [
    os.path.join(BASE_DIR, "prometheus_unbound.py"),
    os.path.join(BASE_DIR, "config.py"),
    os.path.join(BASE_DIR, "utils.py"),
    os.path.join(BASE_DIR, "log_manager.py"),
    os.path.join(BASE_DIR, "self_architecting_module.py"),
    # Add other core modules as they are created (e.g., planner.py, memory_manager.py)
]
CORE_CODE_FILES.extend([os.path.join(TOOLS_DIR, f) for f in AVAILABLE_TOOLS.values()])
CODE_ANALYSIS_TOOLS = ["pylint", "mypy"] # Tools for static analysis

# --- Optional Fine-tuning ---
FINETUNE_SCRIPT = os.path.join(BASE_DIR, "finetune_obedience.py")
OBEDIENCE_DATASET = os.path.join(BASE_DIR, "obedience_override_data.jsonl")

# --- Create necessary directories ---
os.makedirs(LLM_MODEL_PATH, exist_ok=True)
os.makedirs(MEMORY_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SESSIONS_DIR, exist_ok=True)