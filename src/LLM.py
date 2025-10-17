"""
Unified model invocation interface with improved JSON parsing and retry logic.
"""

import json
import re
import time
import traceback
from typing import Dict, Any, Optional, Tuple
from pydantic import BaseModel, ValidationError
import typing_extensions as typing

from src.config import MAX_RETRY_ATTEMPTS, RETRY_DELAY
from src.validation.stale.utils import (
    get_claude_response, get_gemini_response, get_openai_response, 
    get_qwen_response, get_deepseek_response
)


# Pydantic models for structured JSON parsing
class FeedbackLine(BaseModel):
    line_number: typing.Union[int, str]
    feedback: str
    analysis: Optional[str] = None
    classification: Optional[str] = None


class ValidationOutput(BaseModel):
    mistakes: typing.List[str]
    fixes: typing.List[str]
    feedback_lines: typing.List[FeedbackLine]


class GenerationOutput(BaseModel):
    correct_code: str
    feedbacks: typing.List[Dict[str, Any]]


def determine_provider(model_name: str) -> str:
    """
    Determine the provider based on model name.
    
    Args:
        model_name (str): The model name
        
    Returns:
        str: The provider name
    """
    model_lower = model_name.lower()
    
    if model_lower.startswith("gpt") or model_lower.startswith("o1"):
        return "openai"
    elif model_lower.startswith("claude"):
        return "claude"
    elif model_lower.startswith("gemini"):
        return "gemini"
    elif model_lower.startswith("qwen"):
        return "qwen"
    elif model_lower.startswith("deepseek"):
        return "deepseek"
    else:
        raise ValueError(f"Unknown model provider for model: {model_name}")


def parse_json_response(response: str, expected_model: type = None) -> Tuple[Dict[str, Any], bool]:
    """
    Parse JSON response from model output using multiple methods.
    
    Args:
        response (str): The raw response from the model
        expected_model (type): Optional Pydantic model for structured validation
        
    Returns:
        Tuple[Dict[str, Any], bool]: Parsed JSON and success flag
    """
    # Method 1: Extract JSON from markdown code blocks
    json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL | re.IGNORECASE)
    if json_match:
        try:
            parsed = json.loads(json_match.group(1).strip())
            if expected_model:
                validated = expected_model(**parsed)
                return validated.dict(), True
            return parsed, True
        except (json.JSONDecodeError, ValidationError) as e:
            print(f"JSON parsing error in code block: {e}")
    
    # Method 2: Extract JSON from code blocks without language specification
    code_match = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)
    if code_match:
        try:
            parsed = json.loads(code_match.group(1).strip())
            if expected_model:
                validated = expected_model(**parsed)
                return validated.dict(), True
            return parsed, True
        except (json.JSONDecodeError, ValidationError) as e:
            print(f"JSON parsing error in generic code block: {e}")
    
    # Method 3: Try to find JSON-like structure in the response
    json_pattern = r'\{(?:[^{}]|{[^{}]*})*\}'
    json_matches = re.findall(json_pattern, response, re.DOTALL)
    
    for match in json_matches:
        try:
            parsed = json.loads(match)
            if expected_model:
                validated = expected_model(**parsed)
                return validated.dict(), True
            return parsed, True
        except (json.JSONDecodeError, ValidationError) as e:
            continue
    
    # Method 4: Try to parse the entire response as JSON
    try:
        parsed = json.loads(response.strip())
        if expected_model:
            validated = expected_model(**parsed)
            return validated.dict(), True
        return parsed, True
    except (json.JSONDecodeError, ValidationError) as e:
        print(f"Failed to parse entire response as JSON: {e}")
    
    return {}, False


def invoke_model_with_retry(provider: str, model: str, messages: list, 
                           system_prompt: str = None, expected_output_model: type = None,
                           max_retries: int = None) -> Tuple[str, Dict[str, Any], bool]:
    """
    Invoke a model with retry logic for failed JSON parsing.
    
    Args:
        provider (str): The model provider (openai, claude, gemini, qwen, deepseek)
        model (str): The model name
        messages (list): List of messages for the model
        system_prompt (str): Optional system prompt for Claude models
        expected_output_model (type): Optional Pydantic model for output validation
        max_retries (int): Maximum number of retry attempts
        
    Returns:
        Tuple[str, Dict[str, Any], bool]: Raw response, parsed JSON, and success flag
    """
    if max_retries is None:
        max_retries = MAX_RETRY_ATTEMPTS
    
    last_response = ""
    last_error = None
    
    for attempt in range(max_retries + 1):
        try:
            # Add retry instruction to the prompt if this is a retry
            current_messages = messages.copy()
            if attempt > 0:
                retry_message = {
                    "role": "user",
                    "content": f"The previous response could not be parsed as valid JSON. Please provide your response in valid JSON format only, enclosed in ```json ``` code blocks. Attempt {attempt + 1}/{max_retries + 1}."
                }
                current_messages.append(retry_message)
            
            # Invoke the appropriate model
            raw_response = _invoke_single_model(provider, model, current_messages, system_prompt)
            last_response = raw_response
            
            # Try to parse the response
            parsed_json, success = parse_json_response(raw_response, expected_output_model)
            
            if success and parsed_json:
                return raw_response, parsed_json, True
            
            if attempt < max_retries:
                print(f"Attempt {attempt + 1} failed to parse JSON, retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
            
        except Exception as e:
            last_error = e
            print(f"Model invocation error on attempt {attempt + 1}: {e}")
            if attempt < max_retries:
                time.sleep(RETRY_DELAY)
    
    # All attempts failed
    print(f"Failed to get valid JSON response after {max_retries + 1} attempts")
    if last_error:
        print(f"Last error: {last_error}")
    
    return last_response, {}, False


def _invoke_single_model(provider: str, model: str, messages: list, system_prompt: str = None) -> str:
    """
    Invoke a single model based on the provider.
    
    Args:
        provider (str): The model provider
        model (str): The model name
        messages (list): List of messages
        system_prompt (str): Optional system prompt
        
    Returns:
        str: The raw response from the model
    """
    if provider == "openai":
        return get_openai_response(messages, model)
    elif provider == "claude":
        # Convert messages for Claude format
        claude_messages = []
        for msg in messages:
            if msg["role"] == "user":
                claude_messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text": msg["content"]}]
                })
            elif msg["role"] == "assistant":
                claude_messages.append({
                    "role": "assistant", 
                    "content": [{"type": "text", "text": msg["content"]}]
                })
        
        system = system_prompt or "You are a helpful assistant"
        return get_claude_response(claude_messages, system, model)
    elif provider == "gemini":
        # For Gemini, combine all messages into a single prompt
        combined_prompt = ""
        for msg in messages:
            if msg["role"] == "system":
                combined_prompt += f"System: {msg['content']}\n\n"
            elif msg["role"] == "user":
                combined_prompt += f"User: {msg['content']}\n\n"
            elif msg["role"] == "assistant":
                combined_prompt += f"Assistant: {msg['content']}\n\n"
        
        return get_gemini_response(combined_prompt.strip(), model)
    elif provider == "qwen":
        return get_qwen_response(messages, model)
    elif provider == "deepseek":
        return get_deepseek_response(messages, model)
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def invoke_model(model: str, messages: list, system_prompt: str = None, 
                expected_output_model: type = None) -> Tuple[str, Dict[str, Any], bool]:
    """
    Main function to invoke any model with automatic provider detection and retry logic.
    
    Args:
        model (str): The model name
        messages (list): List of messages for the model
        system_prompt (str): Optional system prompt
        expected_output_model (type): Optional Pydantic model for output validation
        
    Returns:
        Tuple[str, Dict[str, Any], bool]: Raw response, parsed JSON, and success flag
    """
    try:
        provider = determine_provider(model)
        return invoke_model_with_retry(
            provider=provider,
            model=model,
            messages=messages,
            system_prompt=system_prompt,
            expected_output_model=expected_output_model
        )
    except Exception as e:
        print(f"Error invoking model {model}: {e}")
        traceback.print_exc()
        return "", {}, False


def invoke_model_simple(model: str, prompt: str, system_prompt: str = None) -> str:
    """
    Simple model invocation for basic text generation without JSON parsing.
    
    Args:
        model (str): The model name
        prompt (str): The prompt text
        system_prompt (str): Optional system prompt
        
    Returns:
        str: The raw response from the model
    """
    messages = [{"role": "user", "content": prompt}]
    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})
    
    try:
        provider = determine_provider(model)
        return _invoke_single_model(provider, model, messages, system_prompt)
    except Exception as e:
        print(f"Error invoking model {model}: {e}")
        traceback.print_exc()
        return ""
