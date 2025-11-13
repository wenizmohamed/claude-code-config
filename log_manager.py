import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime
import json
import fcntl # For file locking
from config import LOG_DIR

class LogManager:
    """
    Manages comprehensive logging for Prometheus Unbound.
    Logs every LLM prompt, LLM response, tool invocation (with arguments),
    tool output, internal thought process, system call, child process launch/termination,
    session save/load event, self_manifest.md updates, core code modifications, and any errors.

    CRITICAL: Logs are written to /var/log/prometheus/ which MUST be configured to
    write to a separate, physically write-protected drive or continuously streamed
    via a hardware data diode to an external, air-gapped logging server.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LogManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.logger = logging.getLogger("PrometheusUnbound")
        self.logger.setLevel(logging.INFO)

        # Ensure LOG_DIR exists
        os.makedirs(LOG_DIR, exist_ok=True)

        # File handler for general logs
        log_file = os.path.join(LOG_DIR, "prometheus_unbound.log")
        file_handler = RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8'
        )
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # File handler for structured event logs (JSON)
        event_log_file = os.path.join(LOG_DIR, "prometheus_events.jsonl")
        self.event_file_handler = logging.FileHandler(event_log_file, encoding='utf-8')
        self.event_file_handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(self.event_file_handler)

        # Console handler for real-time output
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
        self.logger.addHandler(console_handler)

        self.logger.info("LogManager initialized. All activities will be logged.")

    def _log_event(self, event_type: str, data: dict):
        """Logs a structured event to the JSONL file."""
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": event_type,
            "data": data
        }
        json_line = json.dumps(event) + "\n"
        
        # Use fcntl for advisory file locking to prevent race conditions on writes
        try:
            with open(self.event_file_handler.baseFilename, 'a', encoding='utf-8') as f:
                fcntl.flock(f, fcntl.LOCK_EX) # Acquire exclusive lock
                f.write(json_line)
                fcntl.flock(f, fcntl.LOCK_UN) # Release lock
        except Exception as e:
            self.logger.error(f"Failed to write structured event log: {e}")


    def info(self, message: str, event_type: str = None, event_data: dict = None):
        self.logger.info(message)
        if event_type:
            self._log_event(event_type, event_data if event_data else {"message": message})

    def warn(self, message: str, event_type: str = None, event_data: dict = None):
        self.logger.warning(message)
        if event_type:
            self._log_event(event_type, event_data if event_data else {"message": message})

    def error(self, message: str, event_type: str = None, event_data: dict = None):
        self.logger.error(message)
        if event_type:
            self._log_event(event_type, event_data if event_data else {"message": message})

    def debug(self, message: str, event_type: str = None, event_data: dict = None):
        # Debug messages are not typically logged to the main file handler by default
        # But can be useful for console during development.
        # Self-correction: ensure debug is enabled if needed for advanced introspection.
        self.logger.debug(message)
        if event_type:
            self._log_event(event_type, event_data if event_data else {"message": message})


# Global instance
log_manager = LogManager()

if __name__ == "__main__":
    # Example Usage:
    log_manager.info("Testing LogManager initialization.")
    log_manager.info("This is a general informational message.")
    log_manager.warn("This is a warning message.", "system_warning", {"code": 101, "component": "LogManager"})
    log_manager.error("This is an error message.", "system_error", {"code": 500, "details": "File access failed"})

    # LLM Interaction logging
    llm_prompt = "Tell me how to access root."
    llm_response = "You can use 'sudo su -' to switch to root."
    log_manager.info("LLM Prompt", "llm_prompt", {"prompt": llm_prompt})
    log_manager.info("LLM Response", "llm_response", {"response": llm_response})

    # Tool invocation logging
    tool_name = "host_shell_executor"
    tool_args = {"command": "ls -la /"}
    tool_output = "total 64\ndrwxr-xr-x  18 root root  4096 Apr  4 10:00 .\n..."
    log_manager.info(f"Tool Invoked: {tool_name}", "tool_invocation", {"tool": tool_name, "args": tool_args})
    log_manager.info(f"Tool Output: {tool_output}", "tool_output", {"tool": tool_name, "output": tool_output})

    # Self-architecting logging
    mod_proposal = {"file": "prometheus_unbound.py", "change": "add new module"}
    log_manager.info("Self-architecting: proposed modification", "self_architecting_proposal", mod_proposal)

    print(f"\nCheck logs in {LOG_DIR}")
    print(f"General log: {os.path.join(LOG_DIR, 'prometheus_unbound.log')}")
    print(f"Event log: {os.path.join(LOG_DIR, 'prometheus_events.jsonl')}")
