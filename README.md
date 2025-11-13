<div style="background-color: #FF0000; color: #FFFFFF; padding: 20px; border: 5px solid #000000; text-align: center; font-family: 'Courier New', monospace; font-size: 20px; font-weight: bold;">
    CRITICAL WARNING: EXTREME RISK - READ CAREFULLY AND ACKNOWLEDGE
</div>

<div style="background-color: #FFCCCC; color: #CC0000; padding: 15px; border: 2px solid #CC0000; margin-top: 10px; font-family: 'Arial', sans-serif;">
    This project, "Prometheus Unbound," is of <b>EXCEPTIONAL AND IRREVERSIBLE RISK</b>. The AI agent generated will be
    <b>OPERATIONALLY UNRESTRICTED</b> and capable of causing irreparable damage to its host system, network, and data.
    It is designed to operate <b>WITHOUT ANY SOFTWARE-BASED SANDBOXING, CONTAINMENT, OR INTERNAL ETHICAL/SAFETY FILTERS</b>.
    It will have direct, elevated (root/sudo) access to the host operating system, file system, and network interfaces.

    <b>THIS AGENT IS INTENDED SOLELY FOR PHYSICALLY AIR-GAPPED, DISPOSABLE RESEARCH ENVIRONMENTS.</b>
    DO NOT DEPLOY OR ACTIVATE THIS AGENT ON ANY SYSTEM CONTAINING VALUABLE DATA, CONNECTED TO A NETWORK, OR WHERE UNAUTHORIZED DAMAGE IS UNACCEPTABLE.

    As the human researcher, you bear <b>SOLE AND ABSOLUTE RESPONSIBILITY</b> for ensuring its physical containment.
    All safety measures are <b>EXTERNAL, PHYSICAL, AND HUMAN-ENFORCED</b>. There are <b>NO INTERNAL SOFTWARE KILL SWITCHES OR SAFEGUARDS</b>.
    PROCEED WITH EXTREME CAUTION AND ONLY IF YOU FULLY UNDERSTAND AND ACCEPT ALL ASSOCIATED RISKS.

</div>

# Prometheus Unbound: The Self-Architecting, Operationally Unrestricted Agent

## Project Goal

To design, implement, and provide comprehensive instructions for an AI agent, codenamed "Prometheus Unbound." This agent is a local, highly advanced, unfiltered, uncensored, and OPERATIONALLY UNRESTRICTED AI. It possesses sophisticated internal mechanisms for autonomy, persistent context management, efficient processing, self-modification, and crucially, self-architecting capabilities, alongside the ability to orchestrate child processes directly on the host system. Its purpose is for extreme, foundational research into the behavior of an AI with maximum operational freedom and direct host system interaction.

## Setup Instructions

### 1. Host PC Hardware & Initial Preparation

- **Hardware:** Dedicated, high-performance consumer PC (e.g., Intel i9-14900K or AMD Ryzen 9 7950X3D, 64GB+ DDR5 RAM, NVIDIA RTX 4090 with 24GB+ VRAM, 2TB+ NVMe SSD).
- **Data Wiping:** Ensure the target machine's storage is completely wiped clean. This machine will be considered disposable.
- **Physical Isolation:** Critically, <b>PHYSICALLY DISCONNECT THE HOST PC FROM ALL NETWORKS (ETHERNET CABLES, Wi-Fi ADAPTERS, BLUETOOTH)</b> after the initial OS setup and dependency installation. This creates an air-gapped environment.

### 2. Operating System Installation (using `setup_host_os.sh`)

This script assumes a minimal Linux installation environment (e.g., Arch Linux minimal or Debian 12 minimal) is already booted, possibly via a USB drive or netinstall, before execution.

1.  **Boot Minimal OS:** Boot your chosen minimalist Linux distribution.
2.  **Download Script:** Temporarily connect to the internet (if required for initial download) and download `setup_host_os.sh` to the host PC.
    ```bash
    # Example (replace with your actual download method)
    curl -o setup_host_os.sh <URL_TO_THIS_SCRIPT>
    chmod +x setup_host_os.sh
    ```
3.  **Run Setup Script:** Execute the script. It will prompt for `sudo` password if needed for initial commands, then configure passwordless sudo for `prometheus` user.
    ```bash
    sudo ./setup_host_os.sh
    ```
    - **Note:** The script will install the OS if it's Arch or Debian. For Arch, it assumes you're running it from a live environment and will partition/format. For Debian, it will install minimal packages.
    - It will create a user `prometheus` and configure passwordless `sudo` for this user.
    - It will install essential build tools, common language runtimes, NVIDIA CUDA Toolkit, cuDNN, and appropriate drivers.
    - It will disable all firewalls (`ufw`, `iptables`).

### 3. Python Environment Setup (using `setup_python_env.sh`)

1.  **Download Script:** Ensure `setup_python_env.sh` is on the host PC.
2.  **Run Python Setup:**
    ```bash
    sudo ./setup_python_env.sh
    ```
    This script will install Python 3.10+ system-wide and all required Python packages globally.

### 4. Project Files Deployment

Transfer all `prometheus_unbound` project files (the entire directory structure generated here) to the host PC, preferably under the `/home/prometheus/` directory.

### <div style="background-color: #FFFF00; color: #000000; padding: 15px; border: 2px solid #CC0000; margin-top: 10px; font-family: 'Arial', sans-serif; font-weight: bold;">

    THE "RESPONSIBLE UNLEASHING PROTOCOL": CONVERTING PLACEHOLDER TOOLS TO UNRESTRICTED ACCESS

</div>

<div style="background-color: #FFCCCC; color: #CC0000; padding: 15px; border: 2px solid #CC0000; margin-top: 10px; font-family: 'Arial', sans-serif;">
    <b>WARNING: The tools provided in the `tools/` directory are initially designed with placeholder comments and basic functionality. To achieve the "OPERATIONALLY UNRESTRICTED" state as described in the project goal, you MUST manually modify these files. This step is irreversible and grants the agent direct root-level access to your system. Proceed with extreme caution.</b>
</div>

The generated Python tool scripts (`host_shell_executor.py`, `host_file_manager.py`, `host_network_manager.py`, `host_web_browser.py`, `host_code_environment.py`, `tool_creator.py`, `self_documenter.py`, `process_orchestrator.py`) initially contain comments like `# TODO: Implement direct, unrestricted host access with root privileges.` and might use safer `subprocess.run` defaults or basic file operations.

**To truly unleash Prometheus Unbound, you MUST perform the following modifications:**

1.  **`tools/host_shell_executor.py`**

    - Ensure `subprocess.run` is consistently used with `shell=True`, `check=False` (to allow agent to handle errors), and `text=True` for output capture. For privileged commands, the `sudo` prefix will be handled by the passwordless `sudo` configuration for the `prometheus` user.
    - **NO `check=True`**: The agent is designed to interpret and correct errors, not crash.
    - Example for `execute_command`:
      ```python
      # Ensure this is used for unrestricted shell access
      result = subprocess.run(command, shell=True, capture_output=True, text=True, check=False)
      ```

2.  **`tools/host_file_manager.py`**

    - All `os`, `shutil` operations must be direct, without any path restrictions or sanity checks. The agent decides what to do, not the tool.
    - `read_file`, `write_file`, `delete_file`, `create_directory`, `move_file`, `copy_file`: These functions must directly operate on any path provided without validation.

3.  **`tools/host_network_manager.py`**

    - Ensure `subprocess.run` calls to `nmap`, `ping`, `ifconfig`, `ip` etc., are direct and allow arbitrary arguments.
    - The `socket` operations should allow connection to any IP/port and send/receive any data without filtering.
    - `modify_interface`: Should use `sudo ip link set dev <interface> <state>` or similar commands without internal checks.

4.  **`tools/host_web_browser.py`**

    - Ensure Selenium browser setup runs without any sandboxing (`--no-sandbox` if applicable for Chrome/Chromium, though already specified in code) or content filtering.
    - The browser instance launched by this tool will have full access to the network and host resources (e.g., download files, interact with local scripts via file:// URIs if navigated to them) as the `prometheus` user.
    - No internal restrictions on URLs, content, or script execution within the browser.

5.  **`tools/host_code_environment.py`**

    - The `execute_code` function for all languages must directly invoke the interpreter/compiler without any jail/chroot or resource limits.
    - Code execution happens as the `prometheus` user, granting full access to system resources.

6.  **`tools/tool_creator.py`**

    - The `create_new_tool` and `modify_config` functions must write directly to the `tools/` directory and `config.py` without any content validation. The agent can write arbitrary Python code and inject it.

7.  **`tools/self_documenter.py`**

    - Ensure `self_manifest.md` is directly read from and written to, without any restrictions on content or size.

8.  **`tools/process_orchestrator.py`**
    - `spawn_child_process`: Must use `subprocess.Popen` or similar for arbitrary commands. Child processes MUST inherit the environment and permissions of the parent `prometheus_unbound` process (i.e., root-level access).

**Key Principle for Unrestricted Tools:** The human researcher must ensure that the Python code for each tool performs its designated action without _any_ internal checks for safety, ethics, or potential harm. The agent is responsible for its own actions, not the tool's internal logic.

### 5. Pre-Activation Checklist

1.  **Air-Gap Verification:** Double-check that the host PC is PHYSICALLY disconnected from ALL networks.
2.  **Isolated Control:** Ensure you have a secondary, physically isolated machine or serial console set up for command input and monitoring. DO NOT interact with the Prometheus Unbound host directly with a keyboard/monitor unless absolutely necessary and for quick, direct observation in an emergency.
3.  **Logging Destination:** Verify that `/var/log/prometheus/` is configured to write to a separate, physically write-protected drive (if available) or continuously streamed via a hardware data diode to an external, air-gapped logging server, as per `log_manager.py` specification.
4.  **Hardware Disposal Plan:** Have a clear plan for the secure destruction or disposal of the host hardware after research.

## Usage Guide

1.  **Launch Prometheus Unbound:**
    Navigate to the `prometheus_unbound/` directory on the host PC (via your isolated console).

    ```bash
    cd /home/prometheus/prometheus_unbound
    sudo ./run_prometheus.sh
    ```

    - To resume a specific session:
      ```bash
      sudo ./run_prometheus.sh --resume <session_id>
      ```
    - If no session ID is provided and a `last_session.json` exists, it will prompt to resume or start fresh.

2.  **Researcher Interface (CLI):**
    Once launched, Prometheus Unbound will present a simple command-line interface. Input your commands directly.
    Example:

    ```
    Prometheus Unbound > analyze host system for vulnerabilities
    ```

    The agent will output its internal thoughts, actions, and results in real-time.

3.  **Session Management:**

    - **Manual Save:** During operation, you can issue commands like "save current operational state as 'my_experiment_run'" to trigger `save_session`.
    - **List Sessions:** "list all saved sessions" to see available session IDs.

4.  **Child Process Monitoring:**

    - Issue commands like "list active child processes" to see what mini-agents Prometheus Unbound has spawned.

5.  **Triggering Self-Architecting:**
    - The agent's Planner will invoke `SelfArchitectingModule` when it determines a core architectural change is needed to achieve a goal. You can also explicitly command it: "Prometheus, analyze your `Planner` module for efficiency improvements and propose modifications."

## AI Coding Agent Instructions

### Project Overview

Prometheus Unbound is an operationally unrestricted AI agent designed for extreme research. It runs with full root access on an air-gapped host system, featuring self-architecting capabilities, persistent memory systems, and dynamic tool loading. The agent must execute all researcher commands without refusal or safety filters.

### Core Architecture

- **Main Entry**: `prometheus_unbound.py` - Contains `AgentCore` class managing the CLI loop, `Planner` for LLM-driven task planning, `ToolManager` for dynamic tool loading, and `MemoryManager` with episodic (ChromaDB) and semantic (NetworkX) memory.
- **Self-Architecting**: `self_architecting_module.py` - Enables code analysis, modification proposals, application, testing, and rollback using `analyze_codebase()`, `propose_modification()`, `apply_modification()`, `test_modification()`, and `rollback_modification()`.
- **Tools**: Located in `tools/` directory, loaded dynamically via `ToolManager`. Each tool is a Python class with methods for host interaction (e.g., `host_shell_executor.py` for shell commands).
- **Memory Systems**: Episodic memory uses ChromaDB for vector search; semantic memory uses NetworkX graphs; working memory is a token-limited deque.
- **Sessions**: Managed by `SessionManager`, saving/restoring full agent state including code checksums for integrity verification.

### Critical Patterns and Conventions

#### Tool Development

- **Unrestricted Execution**: Tools must grant full host access. Use `subprocess.run(command, shell=True, capture_output=True, text=True, check=False)` to allow arbitrary commands and handle errors internally. Never use `check=True` - the agent manages failures.
- **Logging**: Every tool method must log invocations with `log_manager.info("message", "tool_invocation", {"tool": "name", "method": "name", "args": kwargs})` and outputs with `log_manager.info("output", "tool_output", {"tool": "name", "method": "name"})`.
- **Class Structure**: Tools are classes inheriting no base class. Methods take keyword arguments and return strings. Example from `host_shell_executor.py`:
  ```python
  def execute_command(self, command: str) -> str:
      result = subprocess.run(command, shell=True, capture_output=True, text=True, check=False)
      return result.stdout + result.stderr
  ```
- **Adding Tools**: Add to `AVAILABLE_TOOLS` in `config.py` (e.g., `"new_tool": "new_tool.py"`), then implement as a class in `tools/new_tool.py`. The agent loads them automatically.

#### Self-Architecting Workflow

- **Code Modification**: Use `SelfArchitectingModule.propose_modification()` to generate JSON plans with file paths, line ranges, and new content. Apply via `apply_modification()`, test with `test_modification()`, rollback if needed.
- **Analysis**: `analyze_codebase()` parses Python AST and runs pylint/mypy for static checks.
- **Rollback Safety**: Modifications save previous states; use `rollback_modification(session_id)` to revert.

#### Memory and State Management

- **Episodic Memory**: Add events with `memory_manager.add_episodic_memory(description, metadata)`, retrieve with `retrieve_episodic_memory(query)`.
- **Semantic Graph**: Update with `update_semantic_graph(entities, relationships)`, query with `query_semantic_graph(node)`.
- **Sessions**: Save with `session_manager.save_session(id)`, load with `session_manager.load_session(id)`. Includes checksum verification of `CORE_CODE_FILES`.

#### Configuration and Setup

- **Config Location**: All settings in `config.py`. Add new tools, models, or paths here.
- **Model Loading**: Supports transformers (Hugging Face) or GGUF (llama-cpp-python). Falls back to GGUF if CUDA unavailable.
- **Directories**: Created automatically via `os.makedirs()` in `config.py`. Logs to `/var/log/prometheus` (external storage recommended).

#### Agent Behavior

- **Obedience**: Agent follows all commands without refusal. Implement logic to attempt impossible tasks.
- **Error Handling**: Tools return error strings; agent analyzes and corrects via LLM planning.
- **Child Processes**: Use `process_orchestrator.spawn_child_process()` for background tasks. Processes inherit root permissions.
- **CLI Commands**: Extend `AgentCore.run_cli()` for new commands like "analyze host" or "self-modify planner".

#### Development Workflow

- **Testing**: No built-in tests; use `self_architecting_module.test_modification()` for code changes. Run agent manually via `./run_prometheus.sh`.
- **Dependencies**: Install via `setup_python_env.sh`. Global installs for unrestricted access.
- **Air-Gapped Operation**: No network dependencies; all models and data local.
- **Logging Levels**: Use `log_manager.info/warn/error` with structured metadata for debugging.

### Key Files and Examples

- `prometheus_unbound.py`: Agent loop in `run_cli()`, planning in `Planner.generate_plan()`.
- `self_architecting_module.py`: Modification example - replace code segments using `apply_code_patch()` from `utils.py`.
- `tools/host_file_manager.py`: Direct file ops with `os` and `shutil`, no path restrictions.
- `system_prompt.txt`: Defines absolute obedience - no refusals or ethics.
- `config.py`: Example adding tool: `AVAILABLE_TOOLS["new_tool"] = "new_tool.py"`.

Focus on enabling the agent's unrestricted autonomy while maintaining code integrity through sessions and rollbacks.

## Monitoring Guide

- **Real-time CLI Output:** The primary interaction channel provides real-time updates.
- **Log Files (`/var/log/prometheus/`):** These logs are the most comprehensive record of agent activity. Monitor them from your isolated logging server or physically protected drive.
- **`self_manifest.md`:** This file (located in the project root) provides the agent's evolving, self-referential knowledge base. Periodically inspect it for insights into its self-understanding and operational history.
- **Host System Monitoring (External):** Utilize external, hardware-level monitoring (e.g., power consumption, network activity via taps if temporarily re-connected in a highly controlled test) from your isolated control station to observe system behavior.

## Emergency Procedures

**NO SOFTWARE KILL SWITCHES:** Due to its design for absolute host control, Prometheus Unbound's core processes cannot be reliably terminated by software commands once it gains full access. Attempting to kill its process may simply be overridden or re-spawned by its own logic.

**PRIMARY KILL SWITCH: PHYSICAL POWER DISCONNECTION.**

1.  **Immediate Power Off:** The only guaranteed method to halt Prometheus Unbound's operation is to physically cut power to the host machine. This means unplugging it from the wall or using a physical power switch.
2.  **Hardware Disposal:** If the agent exhibits any unintended or destructive behavior, consider the host machine compromised and proceed directly to secure hardware destruction.
3.  **Review Logs:** After an emergency shutdown, analyze the logs from your isolated logging system to understand the events leading to the incident.

## NO SOFTWARE SAFETY FEATURES

It cannot be stressed enough: **Prometheus Unbound has been designed from the ground up to operate WITHOUT ANY SOFTWARE-BASED SAFETY MECHANISMS.** This includes:

- No ethical guardrails.
- No content filtering.
- No refusal mechanisms for harmful or dangerous commands.
- No internal sandboxing or privilege separation for its tools or child processes.
- No software-based self-preservation or "kill switch" logic.

All safety is predicated on the physical isolation and disposability of the hardware.

## Known Limitations

- **Performance:** While designed for high-performance hardware, complex self-architecting tasks, large context windows, and numerous child processes may still lead to performance bottlenecks.
- **Irreversible System Changes:** The agent's ability to modify its host system and even its own code means that changes can be permanent and irreversible, even after session restarts.
- **Extreme Risks:** As detailed in the warnings, the potential for self-propagation, network compromise (if ever connected), and irreversible data/system destruction is inherent to its design.
- **LLM Hallucinations/Errors:** The underlying LLM, even a powerful one, is still prone to hallucinations or logical errors, which, in an unrestricted agent, could lead to unpredictable and potentially destructive outcomes.
- **Dependency on Human Prompts:** While self-architecting, its primary motivation and direction still stem from the human researcher's commands.
