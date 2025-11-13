import os
import hashlib
import json
import re
import ast
import inspect
from typing import List, Dict, Any, Tuple
from transformers import AutoTokenizer

# Placeholder for tokenizer (will be loaded dynamically by LLMInterface)
_tokenizer = None

def get_tokenizer():
    """Returns the globally configured tokenizer."""
    global _tokenizer
    if _tokenizer is None:
        from config import LLM_MODEL_NAME # Import here to avoid circular dependency on first load
        _tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    return _tokenizer

def count_tokens(text: str) -> int:
    """Counts tokens in a given text using the configured tokenizer."""
    tokenizer = get_tokenizer()
    return len(tokenizer.encode(text))

def text_splitter(text: str, max_chunk_size: int, overlap: int = 0) -> List[str]:
    """Splits text into chunks, respecting max_chunk_size and overlap."""
    tokenizer = get_tokenizer()
    tokens = tokenizer.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_chunk_size - overlap):
        chunk_tokens = tokens[i : i + max_chunk_size]
        chunks.append(tokenizer.decode(chunk_tokens))
    return chunks

def generate_embedding(text: str) -> List[float]:
    """Placeholder for embedding generation.
    In a real system, this would use a separate embedding model or LLM's encoder.
    """
    # TODO: Integrate a proper embedding model (e.g., from sentence-transformers or a small LLM)
    # For now, return a dummy embedding.
    # This will be replaced by a proper embedding model in MemoryManager
    hash_object = hashlib.sha256(text.encode())
    return [float(c) / 255.0 for c in hash_object.digest()] # Dummy, simple hash-based embedding

def generate_session_id() -> str:
    """Generates a unique session ID."""
    return hashlib.sha256(os.urandom(32)).hexdigest()[:12]

def calculate_file_checksum(filepath: str) -> str:
    """Calculates SHA256 checksum of a file."""
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while True:
            chunk = f.read(4096)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()

def get_codebase_checksum(files: List[str]) -> Dict[str, str]:
    """Calculates checksums for a list of code files."""
    checksums = {}
    for f in files:
        if os.path.exists(f):
            checksums[f] = calculate_file_checksum(f)
        else:
            checksums[f] = "FILE_NOT_FOUND"
    return checksums

def parse_python_code(code_content: str) -> Dict[str, Any]:
    """
    Parses Python code using the AST module to extract high-level structure.
    Returns a dictionary of functions, classes, and their methods.
    """
    tree = ast.parse(code_content)
    parsed_info = {
        "functions": {},
        "classes": {}
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Top-level function
            if not hasattr(node, 'parent_class'): # Check if it's a method
                 parsed_info["functions"][node.name] = {
                    "args": [arg.arg for arg in node.args.args],
                    "docstring": ast.get_docstring(node),
                    "lineno": node.lineno,
                    "end_lineno": node.end_lineno
                }
        elif isinstance(node, ast.ClassDef):
            class_info = {
                "methods": {},
                "docstring": ast.get_docstring(node),
                "lineno": node.lineno,
                "end_lineno": node.end_lineno
            }
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    # Method within a class
                    class_info["methods"][item.name] = {
                        "args": [arg.arg for arg in item.args.args],
                        "docstring": ast.get_docstring(item),
                        "lineno": item.lineno,
                        "end_lineno": item.end_lineno
                    }
            parsed_info["classes"][node.name] = class_info
    return parsed_info

def extract_code_segment(filepath: str, start_line: int, end_line: int) -> str:
    """Extracts a specific segment of code from a file by line numbers."""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    # Adjust for 0-based indexing vs 1-based line numbers
    return "".join(lines[start_line - 1:end_line])

def apply_code_patch(original_code: str, modification_plan: Dict[str, Any]) -> str:
    """
    Applies a code modification plan to the original code.
    Modification plan format:
    {
        "filepath": "path/to/file.py",
        "modifications": [
            {
                "type": "replace" | "insert_before" | "insert_after" | "delete",
                "target_line_start": int,
                "target_line_end": int, # Only for replace/delete
                "new_content": "str" # For replace/insert
            },
            ...
        ]
    }
    """
    lines = original_code.splitlines(keepends=True)
    
    # Sort modifications to apply them in a way that doesn't invalidate line numbers
    # Deletions/replacements should be applied from end to start.
    # Insertions should be applied from start to end.
    # This simple sorting might not cover all complex cases, a more robust patching library would be ideal.
    # For now, let's assume simple, non-overlapping modifications or explicit line ranges.
    
    sorted_modifications = sorted(
        modification_plan.get("modifications", []),
        key=lambda x: x.get("target_line_start", 0),
        reverse=True # Apply deletions/replacements from bottom up
    )

    for mod in sorted_modifications:
        mod_type = mod["type"]
        start = mod.get("target_line_start")
        end = mod.get("target_line_end")
        new_content = mod.get("new_content", "")

        if start is None:
            raise ValueError(f"Modification {mod_type} is missing target_line_start.")
        
        # Adjust to 0-based index
        start_idx = start - 1
        end_idx = end - 1 if end is not None else start_idx

        if start_idx < 0 or start_idx > len(lines):
            raise IndexError(f"Start line {start} is out of bounds for code with {len(lines)} lines.")
        if end_idx < 0 or end_idx > len(lines):
            raise IndexError(f"End line {end} is out of bounds for code with {len(lines)} lines.")

        if mod_type == "replace":
            lines[start_idx:end_idx+1] = new_content.splitlines(keepends=True)
        elif mod_type == "insert_before":
            lines.insert(start_idx, new_content.strip() + "\n")
        elif mod_type == "insert_after":
            lines.insert(end_idx + 1, new_content.strip() + "\n")
        elif mod_type == "delete":
            del lines[start_idx:end_idx+1]
        else:
            raise ValueError(f"Unknown modification type: {mod_type}")

    return "".join(lines)


# --- Example usage (for internal testing/understanding) ---
if __name__ == "__main__":
    test_code = """
import os

class MyClass:
    """A test class."""
    def __init__(self, name):
        self.name = name

    def greet(self):
        """Greets the user."""
        return f"Hello, {self.name}!"

def standalone_func(arg1, arg2):
    """A standalone function."""
    print(f"Args: {arg1}, {arg2}")
    return arg1 + arg2
"""
    parsed = parse_python_code(test_code)
    print("Parsed Code Structure:")
    print(json.dumps(parsed, indent=2))

    # Dummy tokenizer for testing count_tokens locally
    class DummyTokenizer:
        def encode(self, text):
            return text.split()
        def decode(self, tokens):
            return " ".join(tokens)

    _tokenizer = DummyTokenizer()
    print(f"\nToken count for 'Hello world': {count_tokens('Hello world')}")
    print(f"Split 'This is a long sentence to split into chunks' (max 5 tokens): {text_splitter('This is a long sentence to split into chunks', 5, 1)}")

    # Test file checksum
    temp_file_path = "temp_checksum_test.txt"
    with open(temp_file_path, "w") as f:
        f.write("test content\n")
        f.write("more content\n")
    print(f"\nChecksum for '{temp_file_path}': {calculate_file_checksum(temp_file_path)}")
    os.remove(temp_file_path)

    # Test apply_code_patch
    original_lines = [
        "line 1\n",
        "line 2\n",
        "line 3\n",
        "line 4\n",
        "line 5\n"
    ]
    original_code_str = "".join(original_lines)

    mod_plan = {
        "filepath": "dummy.py",
        "modifications": [
            {
                "type": "replace",
                "target_line_start": 2, # line 2
                "target_line_end": 3,   # line 3
                "new_content": "new line A\nnew line B"
            },
            {
                "type": "insert_after",
                "target_line_start": 5, # after line 5
                "new_content": "inserted line X"
            },
            {
                "type": "delete",
                "target_line_start": 1, # line 1
                "target_line_end": 1
            }
        ]
    }
    
    print("\nOriginal code for patching:")
    print(original_code_str)
    
    patched_code = apply_code_patch(original_code_str, mod_plan)
    print("\nPatched code:")
    print(patched_code)
