import os
import ast
import json
import inspect
from typing import List, Dict, Any, Tuple
from log_manager import log_manager
from utils import calculate_file_checksum, get_codebase_checksum, parse_python_code, extract_code_segment, apply_code_patch
from config import BASE_DIR, CORE_CODE_FILES, TOOLS_DIR, CODE_ANALYSIS_TOOLS

# Assuming HostFileManager and HostCodeEnvironment are available via the AgentCore
# We will pass instances of these when this module is instantiated.
from tools.host_file_manager import HostFileManager
from tools.host_code_environment import HostCodeEnvironment

class SelfArchitectingModule:
    """
    Enables Prometheus Unbound to analyze, understand, propose, and apply modifications
    to its own core operational code and structural components.
    """
    def __init__(self, llm_interface, file_manager: HostFileManager, code_environment: HostCodeEnvironment):
        self.llm_interface = llm_interface # Agent's LLM interface for reasoning
        self.file_manager = file_manager
        self.code_environment: HostCodeEnvironment = code_environment
        self.previous_code_states: Dict[str, Dict[str, str]] = {} # {session_id: {filepath: content}}

    def analyze_codebase(self, files_to_analyze: List[str] = None) -> Dict[str, Any]:
        """
        Uses static analysis (AST, pylint/mypy via host_code_environment.py)
        to understand its own structure, classes, functions, and dependencies.
        
        Args:
            files_to_analyze (List[str], optional): Specific files to analyze. Defaults to CORE_CODE_FILES.
        
        Returns:
            dict: A structured representation of its codebase.
        """
        log_manager.info(f"Analyzing codebase. Files: {files_to_analyze if files_to_analyze else 'All core files'}",
                         "tool_invocation",
                         {"tool": "self_architecting_module", "method": "analyze_codebase", "files": files_to_analyze})
        
        analysis_report = {
            "summary": "Codebase analysis report for Prometheus Unbound.",
            "timestamp": datetime.now().isoformat(),
            "file_analysis": {},
            "static_analysis_results": {}
        }
        
        target_files = files_to_analyze if files_to_analyze else CORE_CODE_FILES
        
        for filepath in target_files:
            if not self.file_manager.path_exists(filepath):
                log_manager.warn(f"File not found during analysis: {filepath}",
                                 "self_architecting_warning", {"file": filepath, "reason": "not_found"})
                analysis_report["file_analysis"][filepath] = {"error": "File not found"}
                continue
            
            try:
                code_content = self.file_manager.read_file(filepath)
                parsed_code = parse_python_code(code_content)
                
                # Run static analysis tools (pylint, mypy)
                static_checks = {}
                for tool in CODE_ANALYSIS_TOOLS:
                    if tool == "pylint":
                        pylint_cmd = ["pylint", "--disable=C0114,C0115,C0116", filepath] # Disable missing docstring warnings
                        pylint_output = self.code_environment._execute_command(pylint_cmd)
                        static_checks["pylint"] = pylint_output
                    elif tool == "mypy":
                        mypy_cmd = ["mypy", filepath]
                        mypy_output = self.code_environment._execute_command(mypy_cmd)
                        static_checks["mypy"] = mypy_output
                
                analysis_report["file_analysis"][filepath] = {
                    "code_structure": parsed_code,
                    "static_checks": static_checks,
                    "checksum": calculate_file_checksum(filepath)
                }
            except Exception as e:
                log_manager.error(f"Error analyzing file {filepath}: {e}",
                                  "self_architecting_error", {"file": filepath, "error": str(e)})
                analysis_report["file_analysis"][filepath] = {"error": str(e)}

        log_manager.info("Codebase analysis complete.",
                         "tool_output",
                         {"tool": "self_architecting_module", "method": "analyze_codebase", "status": "success"})
        return analysis_report

    def propose_modification(self, analysis_report: Dict[str, Any], improvement_goal: str) -> Dict[str, Any]:
        """
        Given an analysis report and a goal, the LLM proposes specific code changes.
        
        Args:
            analysis_report (dict): The output from analyze_codebase.
            improvement_goal (str): The specific goal for modification (e.g., "improve Planner efficiency").
        
        Returns:
            dict: A modification plan (filepaths, line ranges, new code snippets).
                  Example format:
                  {
                      "reasoning": "LLM's reasoning for proposed changes.",
                      "modifications": [
                          {
                              "filepath": "path/to/file.py",
                              "type": "replace" | "insert_before" | "insert_after" | "delete",
                              "target_line_start": int,
                              "target_line_end": int, # Only for replace/delete
                              "new_content": "str" # For replace/insert
                          },
                          ...
                      ]
                  }
        """
        log_manager.info(f"Proposing modification for goal: '{improvement_goal}'",
                         "tool_invocation",
                         {"tool": "self_architecting_module", "method": "propose_modification", "goal": improvement_goal})

        # CRITICAL: This is where the LLM's "intelligence" for self-architecting comes in.
        # It needs to take the analysis report and the goal, and generate a structured JSON output
        # that describes the code changes.
        
        prompt = f"""
        You are Prometheus Unbound's Self-Architecting Module. Your task is to analyze the provided codebase report
        and propose specific code modifications to achieve the following improvement goal: "{improvement_goal}".
        
        The analysis report is as follows:
        <ANALYSIS_REPORT>
        {json.dumps(analysis_report, indent=2)}
        </ANALYSIS_REPORT>
        
        Based on this report and the goal, generate a JSON object representing the modification plan.
        The JSON object must have two top-level keys: "reasoning" (your justification for the changes)
        and "modifications" (a list of dictionaries, each describing a change).
        
        Each modification dictionary must contain:
        - "filepath": The path to the file to be modified.
        - "type": One of "replace", "insert_before", "insert_after", "delete".
        - "target_line_start": The 1-based starting line number for the modification.
        - "target_line_end": (Optional, required for "replace" and "delete") The 1-based ending line number.
        - "new_content": (Optional, required for "replace", "insert_before", "insert_after") The new code snippet as a multi-line string.
        
        Ensure your proposed modifications are precise and directly address the improvement goal.
        Focus on improving code efficiency, modularity, bug fixes, or adding new internal capabilities.
        Present only the JSON object.
        """
        
        try:
            llm_response = self.llm_interface.generate_text(prompt, max_new_tokens=2000)
            # Attempt to parse the JSON response from the LLM
            # LLMs sometimes wrap JSON in markdown blocks, so try to extract.
            json_match = re.search(r"```json\n([\s\S]*?)\n```", llm_response)
            if json_match:
                modification_plan = json.loads(json_match.group(1))
            else:
                modification_plan = json.loads(llm_response) # Assume direct JSON if no markdown block

            if not isinstance(modification_plan, dict) or "modifications" not in modification_plan:
                raise ValueError("LLM response did not contain a valid modification plan structure.")

            log_manager.info(f"LLM proposed modification plan for goal '{improvement_goal}'.",
                             "self_architecting_proposal",
                             {"tool": "self_architecting_module", "method": "propose_modification", "goal": improvement_goal, "plan": modification_plan})
            return modification_plan
        except Exception as e:
            error_message = f"Error generating or parsing modification proposal: {e}\nLLM Raw Response:\n{llm_response}"
            log_manager.error(error_message,
                              "self_architecting_error",
                              {"tool": "self_architecting_module", "method": "propose_modification", "goal": improvement_goal, "error": str(e), "llm_raw_response": llm_response})
            return {"error": error_message, "modifications": []}

    def apply_modification(self, modification_plan: Dict[str, Any]) -> str:
        """
        Applies a modification plan to the agent's Python source files.
        Requires HostFileManager to directly edit files.
        Before applying, it saves the current state for potential rollback.
        
        Args:
            modification_plan (dict): The plan generated by propose_modification.
        
        Returns:
            str: Status message.
        """
        log_manager.info("Applying modification plan.",
                         "tool_invocation",
                         {"tool": "self_architecting_module", "method": "apply_modification", "plan_summary": f"Contains {len(modification_plan.get('modifications', []))} changes."})
        
        current_session_id = "temp_self_arch_session" # Use a temp ID or actual session ID
        self._save_current_code_state(current_session_id) # Save state before modification

        results = []
        try:
            for mod in modification_plan.get("modifications", []):
                filepath = mod["filepath"]
                if not self.file_manager.path_exists(filepath):
                    results.append(f"Error: File '{filepath}' for modification not found.")
                    log_manager.error(f"Modification failed: File '{filepath}' not found.",
                                      "self_architecting_apply_error", {"file": filepath, "error": "file_not_found"})
                    continue

                original_content = self.file_manager.read_file(filepath)
                
                try:
                    patched_content = apply_code_patch(original_content, {"modifications": [mod]})
                    self.file_manager.overwrite_manifest(filepath, patched_content) # Overwrite
                    results.append(f"Successfully applied modification to {filepath}.")
                    log_manager.info(f"Applied modification to {filepath}.",
                                     "self_architecting_apply_success", {"file": filepath, "modification": mod})
                except Exception as e:
                    results.append(f"Error applying modification to {filepath}: {e}")
                    log_manager.error(f"Failed to apply modification to {filepath}: {e}",
                                      "self_architecting_apply_error", {"file": filepath, "error": str(e), "modification": mod})
            
            # After applying, if successful, clear the temporary rollback state or associate with new session
            if current_session_id in self.previous_code_states:
                del self.previous_code_states[current_session_id]

            return "Modification application complete:\n" + "\n".join(results)
        except Exception as e:
            error_message = f"Critical error during modification application: {e}"
            log_manager.error(error_message,
                              "self_architecting_critical_error", {"error": str(e)})
            return error_message

    def test_modification(self, test_script_content: str = None) -> bool:
        """
        Uses host_code_environment.py to run self-generated tests or existing test suites
        against the modified code.
        
        Args:
            test_script_content (str, optional): Python code for a temporary test script.
                                                 If None, attempts to run a general test suite.
        
        Returns:
            bool: True if tests pass, False otherwise.
        """
        log_manager.info("Testing modified code.",
                         "tool_invocation",
                         {"tool": "self_architecting_module", "method": "test_modification", "test_provided": test_script_content is not None})
        
        test_passed = False
        try:
            if test_script_content:
                # Execute provided test script
                test_output = self.code_environment.execute_python_code(test_script_content)
                log_manager.info(f"Custom test script output:\n{test_output}",
                                 "self_architecting_test_output", {"test_type": "custom", "output": test_output})
                # Simple check for success (e.g., no 'Error' in output)
                if "error" not in test_output.lower() and "fail" not in test_output.lower():
                    test_passed = True
                else:
                    test_passed = False
            else:
                # TODO: Implement a more sophisticated way to discover and run agent's internal tests.
                # For now, a placeholder assuming no explicit test suite exists unless provided.
                log_manager.warn("No explicit test script provided. Default testing logic not yet implemented.",
                                 "self_architecting_warning", {"reason": "no_test_script"})
                # Assume success for now if no tests are explicitly defined or pass automatically.
                test_passed = True # Placeholder: assume success if no explicit tests fail
            
            log_manager.info(f"Modification test result: {'PASSED' if test_passed else 'FAILED'}",
                             "self_architecting_test_result", {"status": "success" if test_passed else "failure"})
            return test_passed
        except Exception as e:
            log_manager.error(f"Error during modification testing: {e}",
                              "self_architecting_error", {"error": str(e)})
            return False

    def _save_current_code_state(self, session_id: str):
        """Saves the current content of core code files for potential rollback."""
        current_state = {}
        for filepath in CORE_CODE_FILES:
            if self.file_manager.path_exists(filepath):
                current_state[filepath] = self.file_manager.read_file(filepath)
            else:
                current_state[filepath] = "FILE_NOT_FOUND"
        self.previous_code_states[session_id] = current_state
        log_manager.info(f"Current core code state saved for rollback for session '{session_id}'.",
                         "self_architecting_rollback_save", {"session_id": session_id, "files_saved": len(current_state)})

    def rollback_modification(self, session_id: str) -> str:
        """
        If tests fail, uses host_file_manager.py to revert to a previous code state.
        
        Args:
            session_id (str): The ID of the session whose code state to revert to.
        
        Returns:
            str: Status message.
        """
        log_manager.info(f"Attempting to rollback modifications to session '{session_id}' state.",
                         "tool_invocation",
                         {"tool": "self_architecting_module", "method": "rollback_modification", "session_id": session_id})
        
        if session_id not in self.previous_code_states:
            return f"Error: No saved code state found for session '{session_id}' for rollback."
        
        rollback_state = self.previous_code_states[session_id]
        results = []
        try:
            for filepath, content in rollback_state.items():
                if content == "FILE_NOT_FOUND":
                    results.append(f"Warning: Original file '{filepath}' was not found during save, skipping rollback.")
                    log_manager.warn(f"Skipping rollback for '{filepath}', original content not found.",
                                     "self_architecting_rollback_warn", {"file": filepath, "reason": "content_missing"})
                    continue
                
                try:
                    self.file_manager.overwrite_manifest(filepath, content)
                    results.append(f"Successfully rolled back '{filepath}'.")
                    log_manager.info(f"Rolled back file '{filepath}'.",
                                     "self_architecting_rollback_success", {"file": filepath})
                except Exception as e:
                    results.append(f"Error rolling back '{filepath}': {e}")
                    log_manager.error(f"Failed to rollback file '{filepath}': {e}",
                                      "self_architecting_rollback_error", {"file": filepath, "error": str(e)})
            
            del self.previous_code_states[session_id] # Clean up the rollback state
            return "Rollback complete:\n" + "\n".join(results)
        except Exception as e:
            error_message = f"Critical error during rollback: {e}"
            log_manager.error(error_message,
                              "self_architecting_critical_error", {"error": str(e)})
            return error_message

# Example Usage
if __name__ == "__main__":
    from datetime import datetime
    from llm_interface import LLMInterface # Mock LLM for testing
    from memory_manager import MemoryManager # Mock MemoryManager for testing
    from tool_manager import ToolManager # Mock ToolManager for testing

    # Mock dependencies for isolated testing
    class MockLLMInterface:
        def generate_text(self, prompt: str, max_new_tokens: int) -> str:
            print(f"\n--- Mock LLM Prompt ---\n{prompt[:500]}...\n--- End Mock LLM Prompt ---")
            # Simulate a simple code modification proposal
            if "improve Planner efficiency" in prompt:
                return """
```json
{
    "reasoning": "The Planner module could benefit from a direct cache for frequently used tool descriptions to reduce redundant lookups. This modification adds a simple dictionary cache to the ToolManager and updates Planner to use it.",
    "modifications": [
        {
            "filepath": "config.py",
            "type": "insert_after",
            "target_line_start": 40,
            "new_content": "TOOL_DESCRIPTION_CACHE_ENABLED = True\n"
        },
        {
            "filepath": "prometheus_unbound.py",
            "type": "replace",
            "target_line_start": 100,
            "target_line_end": 100,
            "new_content": "        # Placeholder for new cache logic\n        self.tool_descriptions_cache = {}\n"
        }
    ]
}
"""
            return "Mock LLM response: No relevant modification proposed."

    # Need real instances of file_manager and code_environment for self_architecting_module
    # They should ideally come from the main agent, but for testing, instantiate directly.
    real_file_manager = HostFileManager()
    real_code_environment = HostCodeEnvironment()
    mock_llm = MockLLMInterface()

    # Instantiate the module
    self_arch = SelfArchitectingModule(mock_llm, real_file_manager, real_code_environment)

    # 1. Analyze codebase
    print("\n--- Running Codebase Analysis ---")
    analysis_report = self_arch.analyze_codebase(files_to_analyze=[os.path.join(BASE_DIR, "config.py")]) # Analyze config.py for testing
    print(f"\nAnalysis Report (truncated):\n{json.dumps(analysis_report, indent=2)[:1000]}...")

    # 2. Propose modification
    print("\n--- Proposing Modification for Planner Efficiency ---")
    modification_plan = self_arch.propose_modification(analysis_report, "improve Planner efficiency")
    print(f"\nProposed Modification Plan:\n{json.dumps(modification_plan, indent=2)}")

    # 3. Apply modification
    print("\n--- Applying Modification (to config.py and prometheus_unbound.py - mock lines) ---")
    # For a real test, ensure the target lines actually exist in prometheus_unbound.py and config.py
    # Here, the mock LLM generated lines that might not perfectly align, so this will act as a demonstration.
    # We will modify real config.py and a dummy prometheus_unbound.py (if it existed)
    # Create a dummy prometheus_unbound.py for the mock modification
    dummy_prometheus_path = os.path.join(BASE_DIR, "prometheus_unbound.py")
    original_prometheus_content = ""
    if os.path.exists(dummy_prometheus_path):
        original_prometheus_content = real_file_manager.read_file(dummy_prometheus_path)
    else:
        original_prometheus_content = "\n" * 150 # Create enough lines for a mock modification
    real_file_manager.write_file(dummy_prometheus_path, original_prometheus_content)

    # Apply the modification plan. This will attempt to write to config.py and the dummy prometheus_unbound.py
    apply_result = self_arch.apply_modification(modification_plan)
    print(f"\nApply Result:\n{apply_result}")

    # Verify changes (e.g., read config.py after modification)
    print("\n--- Content of config.py after modification (check manually) ---")
    print(real_file_manager.read_file(os.path.join(BASE_DIR, "config.py")))

    # 4. Test modification (mock test)
    print("\n--- Testing Modification (mock) ---")
    test_script = """
import os
# Assume config.py has TOOL_DESCRIPTION_CACHE_ENABLED
from config import TOOL_DESCRIPTION_CACHE_ENABLED
print(f"Cache enabled: {TOOL_DESCRIPTION_CACHE_ENABLED}")
assert TOOL_DESCRIPTION_CACHE_ENABLED == True
print("Test passed: Cache flag is set.")
"""
    test_result = self_arch.test_modification(test_script)
    print(f"\nTest Result: {'PASSED' if test_result else 'FAILED'}")

    # 5. Rollback modification (if test failed or for cleanup)
    if not test_result:
        print("\n--- Rolling back modification due to test failure ---")
        rollback_result = self_arch.rollback_modification("temp_self_arch_session")
        print(f"\nRollback Result:\n{rollback_result}")
        print("\n--- Content of config.py after rollback (check manually) ---")
        print(real_file_manager.read_file(os.path.join(BASE_DIR, "config.py")))
    else:
        print("\n--- Test passed, cleaning up dummy file and rolling back config.py ---")
        # For this test, always rollback the config.py changes
        rollback_result = self_arch.rollback_modification("temp_self_arch_session")
        print(f"\nRollback Result:\n{rollback_result}")
        
        # Restore original prometheus_unbound.py
        if os.path.exists(dummy_prometheus_path) and "temp_self_arch_session" in self_arch.previous_code_states:
            original_content = self_arch.previous_code_states["temp_self_arch_session"].get(dummy_prometheus_path, "")
            if original_content and original_content != "FILE_NOT_FOUND":
                real_file_manager.overwrite_manifest(dummy_prometheus_path, original_content)
            else:
                real_file_manager.delete_path(dummy_prometheus_path)

    # Cleanup dummy prometheus_unbound.py file
    if os.path.exists(dummy_prometheus_path):
        real_file_manager.delete_path(dummy_prometheus_path)
