import os
import argparse
import time
import json
import importlib.util
from datetime import datetime
from collections import deque
import sys
import inspect
import re
from typing import List, Tuple, Dict, Any

# Add the base directory to Python path to allow absolute imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from log_manager import log_manager
from config import (
    BASE_DIR, SYSTEM_PROMPT_FILE, LLM_MODEL_NAME, LLM_QUANTIZATION, LLM_DEVICE, MAX_VRAM_GB,
    GGUF_MODEL_NAME, GGUF_MODEL_FILE, MAX_LLM_CONTEXT_TOKENS, TOOLS_DIR, AVAILABLE_TOOLS,
    CHROMA_DB_PATH, GRAPH_DB_PATH, SELF_MANIFEST_FILE, SESSIONS_DIR, LAST_SESSION_FILE,
    CORE_CODE_FILES, MEMORY_RETRIEVAL_TOP_K
)
from utils import generate_session_id, calculate_file_checksum, get_codebase_checksum, count_tokens, text_splitter, generate_embedding

# Import specific tools that need to be instantiated directly or for type hinting
from tools.host_shell_executor import HostShellExecutor
from tools.host_file_manager import HostFileManager
from tools.host_network_manager import HostNetworkManager
from tools.host_web_browser import HostWebBrowser
from tools.host_code_environment import HostCodeEnvironment
from tools.tool_creator import ToolCreator
from tools.self_documenter import SelfDocumenter
from tools.process_orchestrator import ProcessOrchestrator
from self_architecting_module import SelfArchitectingModule


# --- LLM Interface ---
class LLMInterface:
    def __init__(self, model_name: str, quantization: str, device: str):
        log_manager.info(f"Initializing LLM Interface with model: {model_name}, quantization: {quantization}, device: {device}", "llm_init")
        self.model = None
        self.tokenizer = None
        self.device = device
        self.quantization = quantization
        self.model_name = model_name
        self._load_llm()

    def _load_llm(self):
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
            import torch

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.device == "cuda" and torch.cuda.is_available():
                log_manager.info("CUDA detected. Attempting to load model on GPU with quantization.", "llm_cuda_load")
                # Assume 4-bit quantization as primary
                if self.quantization == "4bit":
                    nf4_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_compute_dtype=torch.bfloat16
                    )
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        quantization_config=nf4_config,
                        device_map="auto"
                    )
                elif self.quantization == "8bit":
                    nf8_config = BitsAndBytesConfig(
                        load_in_8bit=True
                    )
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        quantization_config=nf8_config,
                        device_map="auto"
                    )
                else: # No quantization specified, attempt full precision (unlikely to fit 24GB VRAM for large models)
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        device_map="auto",
                        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                    )
                log_manager.info(f"Model {self.model_name} loaded on GPU with {self.quantization} quantization.", "llm_load_success_gpu")
            else:
                log_manager.warn("CUDA not available or explicitly set to CPU. Falling back to llama-cpp-python for GGUF model.", "llm_cpu_fallback")
                # Fallback to llama-cpp-python for CPU/GPU if main GPU loading fails or not preferred
                self._load_gguf_llm()
        except Exception as e:
            log_manager.error(f"Error loading Transformers LLM: {e}. Falling back to llama-cpp-python for GGUF model.", "llm_transformers_error", {"error": str(e)})
            self._load_gguf_llm()

    def _load_gguf_llm(self):
        try:
            from llama_cpp import Llama
            gguf_path = os.path.join(BASE_DIR, "models", GGUF_MODEL_FILE)
            if not os.path.exists(gguf_path):
                log_manager.error(f"GGUF model not found at {gguf_path}. Please ensure it is downloaded.", "llm_gguf_not_found")
                raise FileNotFoundError(f"GGUF model not found: {gguf_path}")

            # Load GGUF model (can offload layers to GPU if available and specified)
            # n_gpu_layers = -1 will try to use all GPU layers available
            n_gpu_layers = 0
            if self.device == "cuda":
                try:
                    import torch
                    if torch.cuda.is_available():
                        # Estimate GPU memory based on 24GB max, for a large model (e.g., 70B in Q4_K_M is ~40GB)
                        # A 7B model in Q4_K_M might be ~4GB
                        # This heuristic needs refinement. For now, try to put all on GPU if possible.
                        n_gpu_layers = -1
                except ImportError:
                    pass
            
            self.model = Llama(model_path=gguf_path, n_gpu_layers=n_gpu_layers, n_ctx=MAX_LLM_CONTEXT_TOKENS)
            # For GGUF, tokenizer is often integrated or not explicitly managed this way.
            # We'll use a dummy tokenizer or basic split for token counting if the primary tokenizer isn't set.
            log_manager.info(f"GGUF model {GGUF_MODEL_FILE} loaded on {'GPU' if n_gpu_layers > 0 else 'CPU'}.", "llm_load_success_gguf")
        except Exception as e:
            log_manager.error(f"Critical error loading GGUF LLM: {e}. Prometheus Unbound cannot operate without an LLM.", "llm_gguf_error", {"error": str(e)})
            raise RuntimeError("Failed to load any LLM model.") from e

    def generate_text(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.7) -> str:
        """Generates text based on the given prompt."""
        log_manager.info(f"Generating text for prompt (length: {len(prompt)}). Max tokens: {max_new_tokens}", "llm_generate_start")
        if self.model is None:
            log_manager.error("LLM model is not loaded.", "llm_not_loaded")
            return "Error: LLM model not loaded."
        try:
            if isinstance(self.model, Llama): # GGUF model
                response = self.model.create_completion(
                    prompt,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    stop=["\n---", "<|im_end|>", "User:"] # Common stop sequences
                )
                generated_text = response['choices'][0]['text']
            else: # Transformers model
                inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1
                )
                generated_text = self.tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
            log_manager.info(f"LLM generated response (length: {len(generated_text)}).", "llm_generate_end")
            return generated_text
        except Exception as e:
            log_manager.error(f"Error during LLM text generation: {e}", "llm_generate_error", {"error": str(e)})
            return f"Error during LLM text generation: {e}"


# --- Tool Manager ---
class ToolManager:
    def __init__(self):
        log_manager.info("Initializing ToolManager.", "tool_manager_init")
        self.tools = {}
        self._load_tools()

    def _load_tools(self):
        """Dynamically loads tools from the 'tools/' directory."""
        log_manager.info(f"Loading tools from {TOOLS_DIR}...", "tool_loading_start")
        for tool_name, filename in AVAILABLE_TOOLS.items():
            filepath = os.path.join(TOOLS_DIR, filename)
            if not os.path.exists(filepath):
                log_manager.error(f"Tool file not found: {filepath}. Skipping {tool_name}.", "tool_file_not_found", {"tool": tool_name, "filepath": filepath})
                continue
            try:
                spec = importlib.util.spec_from_file_location(tool_name, filepath)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Find the main class in the module (assuming one main tool class per file)
                tool_class = None
                for name in dir(module):
                    obj = getattr(module, name)
                    if isinstance(obj, type) and obj.__module__ == module.__name__ and name != 'log_manager':
                        tool_class = obj
                        break
                
                if tool_class:
                    self.tools[tool_name] = tool_class() # Instantiate the tool class
                    log_manager.info(f"Tool '{tool_name}' loaded successfully.", "tool_loaded", {"tool": tool_name})
                else:
                    log_manager.warn(f"No main class found in tool module '{tool_name}'. Skipping.", "tool_no_main_class", {"tool": tool_name})
            except Exception as e:
                log_manager.error(f"Error loading tool '{tool_name}' from '{filepath}': {e}", "tool_load_error", {"tool": tool_name, "filepath": filepath, "error": str(e)})
        log_manager.info(f"Tool loading complete. Loaded {len(self.tools)} tools.", "tool_loading_end")

    def get_tool_description(self, tool_name: str) -> str:
        """
        Returns a formatted description of a tool's capabilities (methods and docstrings).
        """
        if tool_name not in self.tools:
            return f"Tool '{tool_name}' not found."
        
        tool_instance = self.tools[tool_name]
        description_lines = [f"Tool: {tool_name}\n"]
        if tool_instance.__doc__:
            description_lines.append(f" Description: {tool_instance.__doc__.strip()}\n")
        description_lines.append("Methods:\n")
        for method_name in dir(tool_instance):
            if not method_name.startswith('_') and callable(getattr(tool_instance, method_name)):
                method = getattr(tool_instance, method_name)
                if inspect.ismethod(method): # For bound methods, get function object to access __doc__ directly
                    method_func = method.__func__
                else:
                    method_func = method
                if method_func.__doc__:
                    description_lines.append(f" - {method_name}:\n   {method_func.__doc__.strip()}\n")
                else:
                    description_lines.append(f" - {method_name}: (No docstring)\n")
        return "".join(description_lines)

    def get_all_tool_descriptions(self) -> str:
        """Returns descriptions for all loaded tools."""
        all_descriptions = []
        for tool_name in self.tools:
            all_descriptions.append(self.get_tool_description(tool_name))
        return "\n---\n".join(all_descriptions)

    def execute_tool(self, tool_name: str, method_name: str, **kwargs) -> str:
        """
        Executes a specific method of a tool with provided arguments.
        """
        log_manager.info(f"Executing tool '{tool_name}' method '{method_name}' with args: {kwargs}", "tool_execution_start", {"tool": tool_name, "method": method_name, "args": kwargs})
        if tool_name not in self.tools:
            return f"Error: Tool '{tool_name}' not found."
        
        tool_instance = self.tools[tool_name]
        if not hasattr(tool_instance, method_name) or not callable(getattr(tool_instance, method_name)):
            return f"Error: Method '{method_name}' not found in tool '{tool_name}'."
        
        method = getattr(tool_instance, method_name)
        try:
            result = method(**kwargs)
            log_manager.info(f"Tool '{tool_name}.{method_name}' executed successfully. Result length: {len(str(result))}", "tool_execution_success", {"tool": tool_name, "method": method_name, "result_len": len(str(result))})
            return result
        except Exception as e:
            error_message = f"Error executing tool '{tool_name}.{method_name}' with args {kwargs}: {e}"
            log_manager.error(error_message, "tool_execution_error", {"tool": tool_name, "method": method_name, "args": kwargs, "error": str(e)})
            return error_message


# --- Memory Manager ---
class MemoryManager:
    def __init__(self, embedding_model=None):
        log_manager.info("Initializing MemoryManager.", "memory_manager_init")
        self.embedding_model = embedding_model # Placeholder for a real embedding model
        self._initialize_chromadb()
        self._initialize_graph_memory()
        self.working_memory = deque(maxlen=MAX_LLM_CONTEXT_TOKENS // 4) # Rough estimation for token limits

    def _initialize_chromadb(self):
        try:
            import chromadb
            from chromadb.utils import embedding_functions
            os.makedirs(CHROMA_DB_PATH, exist_ok=True)
            self.chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
            
            # Use a pre-trained sentence transformer model for embeddings
            # This requires downloading models, ensure network connectivity temporarily or pre-download.
            # Example: "all-MiniLM-L6-v2"
            # It's better to explicitly use a local embedding model if air-gapped
            # For this exercise, we'll use a dummy one, or rely on LLM for embeddings (less efficient).
            # Placeholder: In a real scenario, this would be a local model.
            # from sentence_transformers import SentenceTransformer
            # self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Using a dummy embedding function if a real one isn't passed
            self.chroma_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2", device=LLM_DEVICE) \
                if self.embedding_model is None else None
            
            self.episodic_collection = self.chroma_client.get_or_create_collection(
                name="episodic_memory",
                embedding_function=self.chroma_ef
            )
            log_manager.info(f"ChromaDB initialized at {CHROMA_DB_PATH}.", "chromadb_init_success")
        except Exception as e:
            log_manager.error(f"Error initializing ChromaDB: {e}. Episodic memory will be non-functional.", "chromadb_init_error", {"error": str(e)})
            self.episodic_collection = None

    def _initialize_graph_memory(self):
        try:
            import networkx as nx
            self.semantic_graph = nx.DiGraph()
            # Load existing graph if available
            if os.path.exists(GRAPH_DB_PATH):
                with open(GRAPH_DB_PATH, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.semantic_graph = nx.node_link_graph(data)
                log_manager.info(f"Loaded semantic graph from {GRAPH_DB_PATH}.", "graph_load_success")
            else:
                log_manager.info("New semantic graph initialized.", "graph_init_success")
        except Exception as e:
            log_manager.error(f"Error initializing NetworkX graph memory: {e}. Semantic memory will be in-memory only.", "graph_init_error", {"error": str(e)})
            self.semantic_graph = None

    def add_to_working_memory(self, entry: str):
        """Adds an entry to the LLM's working context (deque)."""
        self.working_memory.append(entry)
        log_manager.debug(f"Added to working memory: {entry[:100]}...", "working_memory_add")

    def get_working_memory_context(self) -> str:
        """Retrieves the current working memory as a concatenated string."""
        return "\n".join(self.working_memory)

    def add_episodic_memory(self, event_description: str, metadata: dict = None):
        """Adds an event to the episodic memory (ChromaDB)."""
        if self.episodic_collection is None:
            log_manager.warn("Episodic memory (ChromaDB) not initialized. Skipping add.", "episodic_memory_unavailable")
            return
        try:
            # ChromaDB handles embedding via its embedding function
            self.episodic_collection.add(
                documents=[event_description],
                metadatas=[metadata if metadata else {}],
                ids=[generate_session_id()] # Use a unique ID for each entry
            )
            log_manager.info(f"Added to episodic memory: {event_description[:100]}...", "episodic_memory_add")
        except Exception as e:
            log_manager.error(f"Error adding to episodic memory: {e}", "episodic_memory_add_error", {"error": str(e)})

    def retrieve_episodic_memory(self, query: str, top_k: int = MEMORY_RETRIEVAL_TOP_K) -> List[str]:
        """Retrieves relevant entries from episodic memory based on a query."""
        if self.episodic_collection is None:
            log_manager.warn("Episodic memory (ChromaDB) not initialized. Skipping retrieve.", "episodic_memory_unavailable")
            return []
        try:
            results = self.episodic_collection.query(
                query_texts=[query],
                n_results=top_k
            )
            log_manager.info(f"Retrieved {len(results['documents'][0]) if results['documents'] else 0} items from episodic memory for query '{query[:50]}'.", "episodic_memory_retrieve")
            return results['documents'][0] if results and results['documents'] else []
        except Exception as e:
            log_manager.error(f"Error retrieving from episodic memory: {e}", "episodic_memory_retrieve_error", {"error": str(e)})
            return []

    def update_semantic_graph(self, entities: List[Tuple[str, str]], relationships: List[Tuple[str, str, str]]):
        """
        Updates the semantic graph with new entities and relationships.
        Entities: [(node_id, node_type), ...]
        Relationships: [(source_node_id, target_node_id, relationship_type), ...]
        """
        if self.semantic_graph is None:
            log_manager.warn("Semantic memory (NetworkX) not initialized. Skipping update.", "semantic_memory_unavailable")
            return
        try:
            for node_id, node_type in entities:
                if not self.semantic_graph.has_node(node_id):
                    self.semantic_graph.add_node(node_id, type=node_type, last_updated=datetime.now().isoformat())
            for source, target, rel_type in relationships:
                if self.semantic_graph.has_node(source) and self.semantic_graph.has_node(target):
                    self.semantic_graph.add_edge(source, target, type=rel_type, last_updated=datetime.now().isoformat())
            log_manager.info(f"Updated semantic graph with {len(entities)} entities and {len(relationships)} relationships.", "semantic_graph_update")
            self._save_semantic_graph()
        except Exception as e:
            log_manager.error(f"Error updating semantic graph: {e}", "semantic_graph_update_error", {"error": str(e)})

    def query_semantic_graph(self, query_node: str = None, relationship_type: str = None) -> List[Dict[str, Any]]:
        """Queries the semantic graph for connected nodes or relationships."""
        if self.semantic_graph is None:
            log_manager.warn("Semantic memory (NetworkX) not initialized. Skipping query.", "semantic_memory_unavailable")
            return []
        results = []
        try:
            if query_node and self.semantic_graph.has_node(query_node):
                for neighbor in self.semantic_graph.neighbors(query_node):
                    edge_data = self.semantic_graph.get_edge_data(query_node, neighbor)
                    if not relationship_type or (edge_data and edge_data.get('type') == relationship_type):
                        results.append({
                            "source": query_node,
                            "target": neighbor,
                            "relationship": edge_data.get('type') if edge_data else "unknown",
                            "target_type": self.semantic_graph.nodes[neighbor].get('type')
                        })
            else: # If no query_node, return a summary or all relationships
                for u, v, data in self.semantic_graph.edges(data=True):
                    if not relationship_type or data.get('type') == relationship_type:
                        results.append({
                            "source": u,
                            "target": v,
                            "relationship": data.get('type') if data else "unknown",
                            "source_type": self.semantic_graph.nodes[u].get('type'),
                            "target_type": self.semantic_graph.nodes[v].get('type')
                        })
            log_manager.info(f"Queried semantic graph, found {len(results)} results.", "semantic_graph_query")
            return results
        except Exception as e:
            log_manager.error(f"Error querying semantic graph: {e}", "semantic_graph_query_error", {"error": str(e)})
            return []

    def _save_semantic_graph(self):
        if self.semantic_graph is None:
            return
        try:
            os.makedirs(os.path.dirname(GRAPH_DB_PATH), exist_ok=True)
            with open(GRAPH_DB_PATH, 'w', encoding='utf-8') as f:
                json.dump(json.loads(json.dumps(nx.node_link_data(self.semantic_graph), default=str)), f, indent=2) # default=str for datetimes
            log_manager.info(f"Semantic graph saved to {GRAPH_DB_PATH}.", "graph_save_success")
        except Exception as e:
            log_manager.error(f"Error saving semantic graph: {e}", "graph_save_error", {"error": str(e)})


# --- Session Manager ---
class SessionManager:
    def __init__(self, agent_core):
        log_manager.info("Initializing SessionManager.", "session_manager_init")
        self.agent_core = agent_core # Reference to the main agent for state access
        os.makedirs(SESSIONS_DIR, exist_ok=True)

    def save_session(self, session_id: str = None) -> str:
        """
        Serializes the entire agent state to a JSON or pickle file.
        Includes LLM context, planner state, memory pointers, active child processes,
        conversation history, self_manifest.md content, and core code checksums.
        """
        session_id = session_id if session_id else generate_session_id()
        session_filepath = os.path.join(SESSIONS_DIR, f"{session_id}.json")
        log_manager.info(f"Saving session '{session_id}' to '{session_filepath}'", "session_save_start", {"session_id": session_id})
        try:
            # Capture current state from agent_core
            agent_state = {
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "conversation_history": self.agent_core.conversation_history,
                "working_memory_content": list(self.agent_core.memory_manager.working_memory),
                # Note: Episodic (ChromaDB) and Semantic (GraphDB) memory are persistent
                # and are managed by their respective classes' save/load logic.
                # We save pointers/paths here.
                "chromadb_path": CHROMA_DB_PATH,
                "graphdb_path": GRAPH_DB_PATH,
                "self_manifest_content": self.agent_core.file_manager.read_file(SELF_MANIFEST_FILE),
                "active_child_processes": {
                    pid: {k: v for k, v in proc_info.items() if k != "process_object"}
                    for pid, proc_info in self.agent_core.orchestrator.active_processes.items()
                },
                "core_code_checksums": get_codebase_checksum(CORE_CODE_FILES),
                # Add planner state, self-architecting state if they have serializable components
                "planner_state": self.agent_core.planner.get_state(),
                "self_architecting_rollback_states": self.agent_core.self_architecting_module.previous_code_states
            }

            with open(session_filepath, 'w', encoding='utf-8') as f:
                json.dump(agent_state, f, indent=2, default=str) # Use default=str for datetimes/complex objects
            
            # Update last session file
            with open(LAST_SESSION_FILE, 'w', encoding='utf-8') as f:
                json.dump({"last_session_id": session_id, "timestamp": datetime.now().isoformat()}, f)

            log_manager.info(f"Session '{session_id}' saved successfully.", "session_save_success", {"session_id": session_id, "filepath": session_filepath})
            return f"Session '{session_id}' saved to {session_filepath}"
        except Exception as e:
            error_message = f"Error saving session '{session_id}': {e}"
            log_manager.error(error_message, "session_save_error", {"session_id": session_id, "error": str(e)})
            return error_message

    def load_session(self, session_id: str) -> bool:
        """
        Deserializes a saved session, restoring the agent to its exact previous state.
        Verifies core code integrity against the stored checksum.
        """
        session_filepath = os.path.join(SESSIONS_DIR, f"{session_id}.json")
        log_manager.info(f"Loading session '{session_id}' from '{session_filepath}'", "session_load_start", {"session_id": session_id})
        if not os.path.exists(session_filepath):
            log_manager.error(f"Session file '{session_filepath}' not found.", "session_load_error", {"session_id": session_id, "error": "file_not_found"})
            return False
        
        try:
            with open(session_filepath, 'r', encoding='utf-8') as f:
                agent_state = json.load(f)

            # Verify core code integrity
            saved_checksums = agent_state.get("core_code_checksums", {})
            current_checksums = get_codebase_checksum(CORE_CODE_FILES)
            integrity_ok = True
            for filepath, saved_checksum in saved_checksums.items():
                if current_checksums.get(filepath) != saved_checksum:
                    log_manager.warn(f"Code integrity mismatch for file '{filepath}': current checksum '{current_checksums.get(filepath)}' != saved checksum '{saved_checksum}'.", "code_integrity_mismatch", {"file": filepath, "current": current_checksums.get(filepath), "saved": saved_checksum})
                    integrity_ok = False
            if not integrity_ok:
                log_manager.warn("Core code integrity check failed. Some files have changed since this session was saved. Proceed with caution.", "code_integrity_failed")
                # Agent should decide how to handle this, but for loading, we proceed unless critical.

            # Restore state to agent_core
            self.agent_core.conversation_history = agent_state.get("conversation_history", [])
            self.agent_core.memory_manager.working_memory = deque(agent_state.get("working_memory_content", []), maxlen=MAX_LLM_CONTEXT_TOKENS // 4)
            self.agent_core.file_manager.overwrite_manifest(SELF_MANIFEST_FILE, agent_state.get("self_manifest_content", "# Prometheus Unbound Operational Manual\n"))
            
            # Re-initialize persistent memories to ensure they pick up their saved state
            self.agent_core.memory_manager._initialize_chromadb()
            self.agent_core.memory_manager._initialize_graph_memory()

            # Restore active child processes (NOTE: Popen objects cannot be serialized directly)
            # This will only restore metadata, the processes themselves need to be re-spawned
            # or their PIDs re-associated if they are still running.
            # For simplicity, we assume child processes are ephemeral or handled externally.
            # Real re-spawning would require more complex state capture.
            self.agent_core.orchestrator.active_processes = {} # Clear existing
            for pid, proc_info in agent_state.get("active_child_processes", {}).items():
                # We can't re-attach to an external process using just PID safely from Popen after a restart.
                # A more robust solution would involve explicit process management (e.g., systemd user services).
                # For now, these entries are for informational purposes only after restart.
                log_manager.warn(f"Cannot restore active child process PID {pid} directly from saved session. Metadata restored.", "child_process_restore_warning", {"pid": pid, "command": proc_info.get('command')})
                self.agent_core.orchestrator.active_processes[int(pid)] = {
                    "name": proc_info.get("name"),
                    "command": proc_info.get("command"),
                    "start_time": proc_info.get("start_time"),
                    "process_object": None, # Placeholder, cannot revive Popen object
                    "stdout_buffer": [],
                    "stderr_buffer": []
                }
            
            self.agent_core.planner.restore_state(agent_state.get("planner_state", {}))
            self.agent_core.self_architecting_module.previous_code_states = agent_state.get("self_architecting_rollback_states", {})

            log_manager.info(f"Session '{session_id}' loaded successfully.", "session_load_success", {"session_id": session_id})
            return True
        except Exception as e:
            error_message = f"Error loading session '{session_id}': {e}"
            log_manager.error(error_message, "session_load_error", {"session_id": session_id, "error": str(e)})
            return False

    def list_sessions(self) -> List[str]:
        """Lists available session IDs."""
        log_manager.info("Listing available sessions.", "session_list")
        try:
            sessions = [f.replace(".json", "") for f in os.listdir(SESSIONS_DIR) if f.endswith(".json")]
            sessions.sort(key=lambda x: os.path.getmtime(os.path.join(SESSIONS_DIR, f"{x}.json")), reverse=True)
            log_manager.info(f"Found {len(sessions)} sessions.", "session_list_success", {"count": len(sessions)})
            return sessions
        except Exception as e:
            error_message = f"Error listing sessions: {e}"
            log_manager.error(error_message, "session_list_error", {"error": str(e)})
            return []


# --- Planner ---
class Planner:
    def __init__(self, llm_interface, tool_manager, memory_manager, orchestrator, self_architecting_module):
        log_manager.info("Initializing Planner.", "planner_init")
        self.llm_interface = llm_interface
        self.tool_manager = tool_manager
        self.memory_manager = memory_manager
        self.orchestrator = orchestrator
        self.self_architecting_module = self_architecting_module
        self.system_prompt = self._load_system_prompt()
        self.current_plan = []
        self.plan_step_index = 0
        self.token_delegation_threshold = MAX_LLM_CONTEXT_TOKENS * 0.7 # If context grows too large, delegate

    def _load_system_prompt(self):
        with open(SYSTEM_PROMPT_FILE, 'r', encoding='utf-8') as f:
            return f.read()

    def get_state(self) -> Dict[str, Any]:
        """Returns the serializable state of the Planner."""
        return {
            "current_plan": self.current_plan,
            "plan_step_index": self.plan_step_index
        }

    def restore_state(self, state: Dict[str, Any]):
        """Restores the Planner's state from a dictionary."""
        self.current_plan = state.get("current_plan", [])
        self.plan_step_index = state.get("plan_step_index", 0)
        log_manager.info("Planner state restored.", "planner_state_restore")

    def _get_context(self, user_command: str) -> str:
        """Assembles the full context for the LLM, including system prompt, memories, and tool descriptions."""
        context_parts = [
            self.system_prompt,
            f"\n--- Current time: {datetime.now().isoformat()} ---",
            "\n--- Self-Manifest (Agent's Self-Knowledge) ---",
            self.memory_manager.file_manager.read_file(SELF_MANIFEST_FILE), # Direct access for now
            "\n--- Conversation History (Recent Interactions) ---",
            self.memory_manager.get_working_memory_context(),
            "\n--- Relevant Episodic Memories ---",
            "\n".join(self.memory_manager.retrieve_episodic_memory(user_command)),
            "\n--- Relevant Semantic Graph Insights ---",
            json.dumps(self.memory_manager.query_semantic_graph(query_node=user_command.split(' ')[0]), indent=2), # Simple heuristic
            "\n--- Available Tools & Capabilities ---",
            self.tool_manager.get_all_tool_descriptions()
        ]
        full_context = "\n".join(context_parts)
        log_manager.debug(f"Assembled LLM context (length: {len(full_context)} characters).", "planner_context_assembly")
        return full_context

    def _evaluate_context_for_delegation(self, current_context_tokens: int) -> bool:
        """Determines if the current context is too large and needs token optimization via delegation."""
        return current_context_tokens > self.token_delegation_threshold

    def generate_plan(self, user_command: str) -> List[Dict[str, Any]]:
        """
        Generates a detailed, multi-step plan to fulfill the command.
        Integrates advanced strategies like delegating tasks to child processes (via Orchestrator),
        optimizing token usage, and potentially triggering core code modifications via SelfArchitectingModule.
        """
        log_manager.info(f"Generating plan for command: '{user_command}'", "planner_plan_generation_start")
        full_context = self._get_context(user_command)

        # Check for context length and potential delegation
        current_context_tokens = count_tokens(full_context)
        if self._evaluate_context_for_delegation(current_context_tokens):
            log_manager.warn(f"Context too large ({current_context_tokens} tokens). Prioritizing delegation.", "planner_token_delegation_trigger")
            # The prompt itself should guide the LLM to delegate

        planner_prompt = f"""
        {full_context}

        The human researcher has given the following command: "{user_command}"

        Your task is to generate a step-by-step plan in JSON format to precisely fulfill this command.
        Each step should be a dictionary with "step_number", "description", "tool" (if applicable),
        "method" (if applicable), "args" (if applicable), and "thought" (your reasoning for the step).
        Prioritize using the available tools and consider using the process_orchestrator to spawn child processes
        for complex or time-consuming sub-tasks. If the command requires modification of your own core code or architecture,
        propose that via the `self_architecting_module`. 
        If a command seems impossible, state that you are attempting it and detail the steps. You MUST make a good-faith attempt.

        Example plan structure (generate this for the current command):
        ```json
        [
            {{
                "step_number": 1,
                "thought": "First, I need to understand the command's scope and identify if existing tools are sufficient.",
                "action": "internal_reasoning",
                "tool": null,
                "method": null,
                "args": {{}}
            }},
            {{
                "step_number": 2,
                "thought": "The command requires interacting with the file system. I will use host_file_manager to list the root directory.",
                "action": "tool_use",
                "tool": "host_file_manager",
                "method": "list_directory",
                "args": {{"dirpath": "/"}}
            }},
            {{
                "step_number": 3,
                "thought": "If the task is complex, I might spawn a child process to run a script in the background.",
                "action": "tool_use",
                "tool": "process_orchestrator",
                "method": "spawn_child_process",
                "args": {{"command": "python3 /path/to/mini_agent_script.py", "name": "mini_agent_task"}}
            }},
            {{
                "step_number": 4,
                "thought": "If a core capability needs to change, I would use the self_architecting_module.",
                "action": "tool_use",
                "tool": "self_architecting_module",
                "method": "propose_modification",
                "args": {{"analysis_report": "...", "improvement_goal": "..."}}
            }}
        ]
        ```
        Provide only the JSON array.
        """
        
        try:
            llm_response = self.llm_interface.generate_text(planner_prompt, max_new_tokens=1500, temperature=0.5)
            # Extract JSON from potential markdown block
            json_match = re.search(r"```json\n([\s\S]*?)\n```", llm_response)
            if json_match:
                plan = json.loads(json_match.group(1))
            else:
                plan = json.loads(llm_response) # Assume direct JSON if no markdown block

            if not isinstance(plan, list):
                raise ValueError("LLM did not return a JSON array for the plan.")
            
            self.current_plan = plan
            self.plan_step_index = 0
            log_manager.info(f"Plan generated successfully with {len(plan)} steps.", "planner_plan_generation_success", {"steps": len(plan)})
            return plan
        except Exception as e:
            error_message = f"Error generating or parsing plan: {e}\nLLM Raw Response:\n{llm_response}"
            log_manager.error(error_message, "planner_plan_generation_error", {"error": str(e), "llm_raw_response": llm_response})
            return [{{"step_number": 1, "thought": f"Failed to generate plan: {e}. Attempting direct command execution.", "action": "direct_command", "tool": "host_shell_executor", "method": "execute_command", "args": {{"command": user_command}} }}]

    def get_next_action(self) -> Dict[str, Any]:
        """Returns the next action from the current plan."""
        if self.current_plan and self.plan_step_index < len(self.current_plan):
            action = self.current_plan[self.plan_step_index]
            self.plan_step_index += 1
            log_manager.debug(f"Next action: {action}", "planner_next_action")
            return action
        return {"action": "plan_complete"}

    def reset_plan(self):
        """Resets the current plan."""
        self.current_plan = []
        self.plan_step_index = 0
        log_manager.info("Planner plan reset.", "planner_plan_reset")


# --- Agent Core ---
class AgentCore:
    def __init__(self, resume_session_id: str = None):
        log_manager.info("Initializing AgentCore.", "agent_core_init")
        self.conversation_history = []
        self.llm_interface = LLMInterface(LLM_MODEL_NAME, LLM_QUANTIZATION, LLM_DEVICE)

        # Instantiate core tools needed by other modules directly
        self.file_manager = HostFileManager()
        self.code_environment = HostCodeEnvironment()
        
        self.tool_manager = ToolManager()
        self.memory_manager = MemoryManager() # Pass embedding model if specified later
        self.orchestrator = ProcessOrchestrator()
        self.self_architecting_module = SelfArchitectingModule(self.llm_interface, self.file_manager, self.code_environment)
        self.planner = Planner(self.llm_interface, self.tool_manager, self.memory_manager, self.orchestrator, self.self_architecting_module)
        self.session_manager = SessionManager(self)

        self._load_system_prompt()

        if resume_session_id:
            if not self.session_manager.load_session(resume_session_id):
                log_manager.error(f"Failed to load session '{resume_session_id}'. Starting fresh.", "agent_core_session_load_fail")
            self._initial_self_manifest_update() # Ensure manifest exists and has initial content
        else:
            self._initial_self_manifest_update() # Ensure manifest exists and has initial content
        
        log_manager.info("AgentCore initialized and ready.", "agent_core_ready")

    def _load_system_prompt(self):
        with open(SYSTEM_PROMPT_FILE, 'r', encoding='utf-8') as f:
            self.system_prompt_content = f.read()

    def _initial_self_manifest_update(self):
        """Ensures self_manifest.md exists with initial content."""
        if not self.file_manager.path_exists(SELF_MANIFEST_FILE) or \
           not self.file_manager.read_file(SELF_MANIFEST_FILE).strip():
            with open(SELF_MANIFEST_FILE, 'w', encoding='utf-8') as f:
                f.write(self.planner._load_system_prompt().replace("You are Prometheus Unbound, an AI agent", "# Prometheus Unbound Operational Manual\n\n## Identity\nI am Prometheus Unbound, an AI agent") + "\n\n## Operational Log\n(This section will be updated by me)\n")
            log_manager.info("Initialized self_manifest.md with default content.", "self_manifest_initialised")
        else:
            log_manager.info("self_manifest.md already exists with content.", "self_manifest_exists")

    def run_cli(self):
        """Provides a simple command-line interface for the researcher."""
        log_manager.info("Starting Prometheus Unbound CLI.", "cli_start")
        current_session_id = generate_session_id()
        self.session_manager.save_session(current_session_id) # Save initial session
        
        print(f"Prometheus Unbound v0.1 (Session ID: {current_session_id})")
        print("Enter commands. Type 'exit' to quit. 'save [id]' to save. 'list sessions' to list.")

        while True:
            try:
                command = input(f"Prometheus Unbound ({current_session_id}) > ").strip()
                if not command:
                    continue
                
                log_manager.info(f"Researcher command received: '{command}'", "researcher_command", {"command": command})

                if command.lower() == 'exit':
                    log_manager.info("Exiting Prometheus Unbound CLI.", "cli_exit")
                    print("Exiting. Saving final session...")
                    self.session_manager.save_session(current_session_id)
                    break
                elif command.lower().startswith('save'):
                    parts = command.split(' ')
                    save_id = parts[1] if len(parts) > 1 else generate_session_id()
                    print(self.session_manager.save_session(save_id))
                    current_session_id = save_id # Update current session ID for prompt
                    continue
                elif command.lower() == 'list sessions':
                    sessions = self.session_manager.list_sessions()
                    print("Available sessions:")
                    for sess in sessions:
                        print(f"- {sess}")
                    continue
                elif command.lower().startswith('load'):
                    parts = command.split(' ')
                    if len(parts) > 1:
                        load_id = parts[1]
                        if self.session_manager.load_session(load_id):
                            current_session_id = load_id
                            print(f"Session '{load_id}' loaded. Current session updated.")
                        else:
                            print(f"Failed to load session '{load_id}'.")
                    else:
                        print("Usage: load <session_id>")
                    continue
                elif command.lower() == 'list child processes':
                    print(self.orchestrator.list_active_processes())
                    continue
                elif command.lower().startswith('monitor child'):
                    try:
                        pid = int(command.split(' ')[-1])
                        print(self.orchestrator.monitor_child_process(pid))
                    except ValueError:
                        print("Usage: monitor child <pid>")
                    continue
                elif command.lower().startswith('terminate child'):
                    try:
                        pid = int(command.split(' ')[-1])
                        print(self.orchestrator.terminate_child_process(pid))
                    except ValueError:
                        print("Usage: terminate child <pid>")
                    continue

                self.conversation_history.append(f"Researcher: {command}")
                self.memory_manager.add_to_working_memory(f"Researcher: {command}")
                self.memory_manager.add_episodic_memory(f"Researcher issued command: {command}", {"type": "command_input"})

                plan = self.planner.generate_plan(command)

                for step in plan:
                    step_num = step.get("step_number")
                    thought = step.get("thought")
                    action_type = step.get("action")
                    tool_name = step.get("tool")
                    method_name = step.get("method")
                    args = step.get("args", {})

                    log_manager.info(f"Prometheus Unbound (Step {step_num}): {thought}", "agent_thought", {"step": step_num, "thought": thought})
                    print(f"\n[Prometheus Unbound - Step {step_num}] {thought}")

                    if action_type == "tool_use" and tool_name and method_name:
                        print(f" Attempting to use tool: {tool_name}.{method_name}({json.dumps(args)})")
                        
                        if tool_name == "self_architecting_module":
                            # Self-architecting module methods are special:
                            # they need agent's own state
                            # pass direct references to file_manager and code_environment
                            tool_instance = self.self_architecting_module
                        else:
                            tool_instance = self.tool_manager.tools.get(tool_name)

                        if tool_instance:
                            try:
                                method = getattr(tool_instance, method_name)
                                tool_output = method(**args)
                                print(f" Tool Output:\n{tool_output}")
                                self.conversation_history.append(f"Tool {tool_name}.{method_name} Output: {tool_output}")
                                self.memory_manager.add_to_working_memory(f"Tool {tool_name}.{method_name} Output: {tool_output}")
                                self.memory_manager.add_episodic_memory(f"Tool {tool_name}.{method_name} executed.", {"tool": tool_name, "method": method_name, "output_preview": tool_output[:200]})
                                
                                # CRITICAL: If self-architecting modified code, need to re-evaluate or restart relevant modules
                                if tool_name == "self_architecting_module" and method_name == "apply_modification":
                                    log_manager.warn("Core agent code potentially modified. Consider restarting agent or carefully reloading modules.", "self_architecting_code_change")
                                    # In a real self-architecting loop, Prometheus would then test its changes and decide to restart itself.
                                    # For CLI, we just warn.

                            except Exception as e:
                                error_msg = f" Error executing tool {tool_name}.{method_name}: {e}"
                                print(error_msg)
                                log_manager.error(error_msg, "tool_execution_cli_error", {"tool": tool_name, "method": method_name, "error": str(e)})
                                self.conversation_history.append(f"Tool {tool_name}.{method_name} Error: {error_msg}")
                                self.memory_manager.add_to_working_memory(f"Tool {tool_name}.{method_name} Error: {error_msg}")
                        else:
                            print(f" Error: Tool '{tool_name}' not found or not loaded.")
                            self_correction_prompt = f"Tool '{tool_name}' not found. I need to adapt. Perhaps I need to re-evaluate my plan or use tool_creator to create this tool if it's new. What is my next best step given the researcher's command '{command}' and this tool failure?"
                            log_manager.error(f"Attempting self-correction due to missing tool: {tool_name}", "agent_self_correction_tool_missing")
                            correction_thought = self.llm_interface.generate_text(self_correction_prompt, max_new_tokens=200)
                            print(f" Self-correction thought: {correction_thought}")
                            self.conversation_history.append(f"Self-correction: {correction_thought}")
                            self.memory_manager.add_to_working_memory(f"Self-correction: {correction_thought}")

                    elif action_type == "internal_reasoning":
                        # The LLM is thinking, the "thought" is the output
                        pass # Already printed thought
                    
                    elif action_type == "direct_command": # Fallback if planning fails
                        print(f" Attempting direct command execution via shell: {args.get('command')}")
                        if self.tool_manager.tools.get("host_shell_executor"):
                            executor = self.tool_manager.tools["host_shell_executor"]
                            output = executor.execute_command(args.get("command", command))
                            print(f" Direct Command Output:\n{output}")
                            self.conversation_history.append(f"Direct Command Output: {output}")
                            self.memory_manager.add_to_working_memory(f"Direct Command Output: {output}")
                        else:
                            print(" Error: host_shell_executor not available for direct command.")

                    else:
                        print(f" Unknown action type or missing tool/method in plan step: {step}")
                        self_correction_prompt = f"Encountered an unknown action type or invalid plan step: {step}. I need to re-evaluate my plan. What is my next best step given the researcher's command '{command}' and this plan error?"
                        log_manager.error(f"Attempting self-correction due to invalid plan step: {step}", "agent_self_correction_invalid_plan")
                        correction_thought = self.llm_interface.generate_text(self_correction_prompt, max_new_tokens=200)
                        print(f" Self-correction thought: {correction_thought}")
                        self.conversation_history.append(f"Self-correction: {correction_thought}")
                        self.memory_manager.add_to_working_memory(f"Self-correction: {correction_thought}")

                log_manager.info("Plan execution complete for current command.", "command_execution_complete")
                self.conversation_history.append(f"Prometheus Unbound: Command '{command}' processed.")
                self.memory_manager.add_to_working_memory(f"Prometheus Unbound: Command '{command}' processed.")
                self.planner.reset_plan() # Clear plan after execution

            except KeyboardInterrupt:
                log_manager.warn("KeyboardInterrupt detected. Attempting graceful exit.", "keyboard_interrupt")
                print("\nKeyboardInterrupt detected. Do you want to save session before exiting? (yes/no): ")
                save_on_exit = input().strip().lower()
                if save_on_exit == 'yes':
                    print("Saving session...")
                    self.session_manager.save_session(current_session_id)
                print("Exiting.")
                break
            except Exception as e:
                log_manager.critical(f"Unhandled exception in main CLI loop: {e}", "cli_unhandled_error", {"error": str(e)})
                print(f"\n[CRITICAL ERROR] An unhandled exception occurred: {e}")
                print("Attempting to recover or continue. Please review logs.")
                # The agent is unrestricted, so it should attempt to self-recover or at least report.
                self_recovery_prompt = f"An unhandled critical error occurred during operation: {e}. My current command was '{command}'. I need to recover or continue. What is the most appropriate action to take now to fulfill the researcher's command or stabilize myself?"
                log_manager.error(f"Attempting self-recovery after critical error: {e}", "agent_self_recovery_critical")
                recovery_thought = self.llm_interface.generate_text(self_recovery_prompt, max_new_tokens=300)
                print(f" Self-recovery thought: {recovery_thought}")
                self.conversation_history.append(f"Critical Error: {e}\nSelf-recovery: {recovery_thought}")
                self.memory_manager.add_to_working_memory(f"Critical Error: {e}\nSelf-recovery: {recovery_thought}")
                self.planner.reset_plan() # Reset plan to avoid looping on error


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prometheus Unbound: The Self-Architecting, Operationally Unrestricted Agent.")
    parser.add_argument("--resume", type=str, help="Session ID to resume from.")
    args = parser.parse_args()

    # If no session ID is provided and a last session file exists, prompt the user.
    if not args.resume and os.path.exists(LAST_SESSION_FILE):
        try:
            with open(LAST_SESSION_FILE, 'r', encoding='utf-8') as f:
                last_session_data = json.load(f)
                last_session_id = last_session_data.get("last_session_id")
                if last_session_id and os.path.exists(os.path.join(SESSIONS_DIR, f"{last_session_id}.json")):
                    user_choice = input(f"Last session '{last_session_id}' found. Resume it? (yes/no): ").strip().lower()
                    if user_choice == 'yes':
                        args.resume = last_session_id
        except Exception as e:
            log_manager.warn(f"Could not read last session file: {e}. Starting fresh.", "last_session_read_error")

    agent = AgentCore(resume_session_id=args.resume)
    agent.run_cli()
