"""
Prompt templates for model invocation
"""

import json
import os
from typing import Optional, Dict, Any, List

from src.config import NUM_SIDS
from src.generation.data import load_existing_results
from src.generation.models import GeneratorData
from src.utils import print_warning, print_error

def generator_data_failure(sid: int, error: str) -> 'GeneratorData':
    """
    Create a failure GeneratorData object for a given SID.
    
    Args:
        sid (int): The SID to process
        error (str): Error message for the failure
        
    Returns:
        GeneratorData: Result with SID, error message, and success status
    """
    return GeneratorData(
        sid=sid,
        error=error,
        generator_data=None,
        success=False
    )


def transform_llm_output_to_generator_data(llm_output: Dict[str, Any], sid: int, pid: int, category_required: bool = False) -> Optional[Dict[str, Any]]:
    """
    Transform LLM output to the complete Generator data format.
    
    Args:
        llm_output: Raw LLM output in format {"correct_code": str, "feedbacks": [...]}
        sid: Student ID
        pid: Problem ID
        category_required: Whether category field is required in feedback
        
    Returns:
        Dict or None: Complete generator data structure if successful, None otherwise
    """
    from .models import convert_llm_to_generator_data, load_student_code_mapping
    
    # Load student code mapping (cached after first call)
    if not hasattr(transform_llm_output_to_generator_data, '_student_code_mapping'):
        transform_llm_output_to_generator_data._student_code_mapping = load_student_code_mapping()
    
    return convert_llm_to_generator_data(
        llm_output, 
        sid, 
        pid, 
        transform_llm_output_to_generator_data._student_code_mapping,
        category_required
    )


def validate_and_save_generator_data(data: Dict[str, Any], output_path: str, category_required: bool = True) -> bool:
    """
    Validate Generator data and save to file if valid.
    
    Args:
        data: Generator data to validate and save
        output_path: Path to save the data
        category_required: Whether category field is required in feedback
        
    Returns:
        bool: True if data was valid and saved, False otherwise
    """
    from .models import validate_generator_data
    
    if validate_generator_data(data, category_required):
        try:
            # Load existing data
            existing_results = load_existing_results(output_path)
            
            # Add new data
            existing_results.append(data)
            
            # Save updated data
            with open(output_path, 'w') as f:
                json.dump(existing_results, f, indent=4)
            
            print(f"Successfully saved data for SID {data['sid']}")
            return True
            
        except Exception as e:
            print_error(f"Error saving data for SID {data['sid']}: {e}")
            return False
    else:
        print_error(f"Data validation failed for SID {data['sid']}")
        return False

