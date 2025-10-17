"""
Utility functions for the project.
"""

import os
import json
import datetime
from typing import Any, Dict, List, Optional


def print_warning(message: str) -> None:
    """Print warning message with orange color formatting."""
    print(f"⚠️\t\033[38;5;208m{message}\033[0m")


def print_error(message: str) -> None:
    """Print error message with red color formatting."""
    print(f"❌\t\033[91m{message}\033[0m")


def print_success(message: str) -> None:
    """Print success message with green color formatting."""
    print(f"✅\t\033[92m{message}\033[0m")


def print_info(message: str) -> None:
    """Print info message with blue color formatting."""
    print(f"ℹ️\t\033[94m{message}\033[0m")


def ensure_directory_exists(directory_path: str) -> str:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory_path: Path to the directory
        
    Returns:
        str: The directory path
    """
    os.makedirs(directory_path, exist_ok=True)
    return directory_path


def load_json_file(file_path: str) -> Optional[Any]:
    """
    Safely load a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        The loaded JSON data or None if failed
    """
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print_error(f"Error loading JSON file {file_path}: {e}")
        return None


def save_json_file(data: Any, file_path: str, indent: int = 4) -> bool:
    """
    Safely save data to a JSON file.
    
    Args:
        data: Data to save
        file_path: Path to save the file
        indent: JSON indentation level
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Ensure directory exists
        directory = os.path.dirname(file_path)
        if directory:
            ensure_directory_exists(directory)
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=indent)
        return True
    except Exception as e:
        print_error(f"Error saving JSON file {file_path}: {e}")
        return False


def format_timestamp(timestamp: Optional[str] = None) -> str:
    """
    Format a timestamp or create a new one.
    
    Args:
        timestamp: Optional existing timestamp
        
    Returns:
        str: Formatted timestamp
    """
    if timestamp:
        return timestamp
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def safe_get_nested_value(data: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    """
    Safely get a nested value from a dictionary.
    
    Args:
        data: Dictionary to search
        keys: List of keys representing the path
        default: Default value if key path not found
        
    Returns:
        The value at the key path or default
    """
    try:
        current = data
        for key in keys:
            current = current[key]
        return current
    except (KeyError, TypeError, IndexError):
        return default


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace in text for comparison purposes.
    
    Args:
        text: Text to normalize
        
    Returns:
        str: Text with normalized whitespace
    """
    return ' '.join(text.split())


def truncate_string(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate a string if it exceeds the maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum allowed length
        suffix: Suffix to add when truncating
        
    Returns:
        str: Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def validate_model_name(model_name: str, valid_models: List[str]) -> bool:
    """
    Validate that a model name is in the list of valid models.
    
    Args:
        model_name: Name of the model to validate
        valid_models: List of valid model names
        
    Returns:
        bool: True if valid, False otherwise
    """
    return model_name in valid_models

# TODO: This is quite approximative and should be improved
def extract_error_type(error_message: str) -> str:
    """
    Extract error type from an error message.
    
    Args:
        error_message: Error message to analyze
        
    Returns:
        str: Categorized error type
    """
    error_message_lower = error_message.lower()
    
    if "429" in error_message or "quota" in error_message_lower:
        return "quota_exceeded"
    elif "timeout" in error_message_lower:
        return "timeout"
    elif "connection" in error_message_lower:
        return "connection_error"
    elif "json" in error_message_lower or "parse" in error_message_lower:
        return "parse_error"
    elif "authentication" in error_message_lower or "unauthorized" in error_message_lower:
        return "auth_error"
    elif "not found" in error_message_lower or "404" in error_message:
        return "not_found"
    else:
        return "unknown_error"


def get_file_extension(file_path: str) -> str:
    """
    Get the file extension from a file path.
    
    Args:
        file_path: Path to the file
        
    Returns:
        str: File extension (without the dot)
    """
    return os.path.splitext(file_path)[1].lstrip('.')


def is_valid_json_string(text: str) -> bool:
    """
    Check if a string is valid JSON.
    
    Args:
        text: String to check
        
    Returns:
        bool: True if valid JSON, False otherwise
    """
    try:
        json.loads(text)
        return True
    except (json.JSONDecodeError, TypeError):
        return False
