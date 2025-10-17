import os
from jsonschema import ValidationError
import pandas as pd
from typing import Dict, List, Union, Any, Optional
import json
from src.config import MODELS_GEN, NUM_SIDS, VALIDATOR_REPAIR
from src.generation.models import GenerationBatch, GeneratorData
from src.utils import print_warning, print_error


# Configuration constants
DEFAULT_QUERIES_FILE_PATH = os.path.join('..', '..', 'data', 'prompts', 'gaide_queries.json')
DEFAULT_ENCODING = 'utf-8'

# Global cache for GAIED queries
_gaied_queries_cache: Optional[List[Dict[str, Any]]] = None

def load_gaied_queries(file_path: str = None) -> List[Dict[str, Any]]:
    """
    Load GAIED queries from JSON file with caching.
    
    Args:
        file_path (str, optional): Path to gaide_queries.json file. 
                                 If None, uses default relative path.
    
    Returns:
        List[Dict[str, Any]]: List of query objects with 'sid' and 'prompt' keys
    
    Raises:
        FileNotFoundError: If the queries file cannot be found
        json.JSONDecodeError: If the JSON file is malformed
    """
    global _gaied_queries_cache
    
    # Return cached data if available
    if _gaied_queries_cache is not None:
        return _gaied_queries_cache
    
    # Determine file path
    if file_path is None:
        # Default path relative to src directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, DEFAULT_QUERIES_FILE_PATH)
    
    # Load and cache the data
    try:
        with open(file_path, 'r', encoding=DEFAULT_ENCODING) as f:
            _gaied_queries_cache = json.load(f)
        
        assert len(_gaied_queries_cache) == NUM_SIDS, \
            f"Expected {NUM_SIDS} queries, but found {len(_gaied_queries_cache)} in {file_path}"
        
        return _gaied_queries_cache
        
    except FileNotFoundError:
        print_warning(f"GAIED queries file not found at {file_path}")
        _gaied_queries_cache = []
        return _gaied_queries_cache
    except json.JSONDecodeError as e:
        print_error(f"Invalid JSON in GAIED queries file: {e}")
        raise

def get_prompt_by_sid(sid: int, file_path: str = None) -> Optional[str]:
    """
    Retrieve a prompt from GAIED queries by SID.
    
    Args:
        sid (int): The SID (Student ID) to search for
        file_path (str, optional): Path to gaide_queries.json file
    
    Returns:
        Optional[str]: The prompt text if found, None otherwise
    
    Example:
        >>> prompt = get_prompt_by_sid(1)
        >>> if prompt:
        ...     print(f"Found prompt for SID 1: {len(prompt)} characters")
    """
    queries = load_gaied_queries(file_path)
    
    for query in queries:
        if query.get('sid') == sid:
            return query.get('prompt')
    
    print_warning(f"No prompt found for SID {sid}")
    return None

def load_student_code_mapping() -> Dict[int, str]:
    """
    Load student code mapping from the Excel dataset.
    
    Returns:
        Dict[int, str]: Mapping of sid to student_code
    """
    try:
        # Path to the dataset relative to src/generation
        dataset_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'GAIED', 'dataset.xlsx')
        
        # Read the buggy_submissions sheet
        buggy_submissions = pd.read_excel(dataset_path, sheet_name='buggy_submissions')
        
        # Create mapping of sid to code
        student_code_mapping = {}
        for _, row in buggy_submissions.iterrows():
            sid = row['sid']
            code = row['code']
            student_code_mapping[sid] = code
        
        assert len(student_code_mapping) == NUM_SIDS, \
            f"Expected {NUM_SIDS} student codes, but found {len(student_code_mapping)} in {dataset_path}"
        
        return student_code_mapping
        
    except Exception as e:
        print_warning(f"Could not load student code mapping: {e}")
        return {}

def load_pid_mapping() -> Dict[int, int]:
    """
    Load problem ID mapping from the Excel dataset.
    
    Returns:
        Dict[int, int]: Mapping of sid to pid
    """
    try:
        # Path to the dataset relative to src/generation
        dataset_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'GAIED', 'dataset.xlsx')
        
        # Read the buggy_submissions sheet
        buggy_submissions = pd.read_excel(dataset_path, sheet_name='buggy_submissions')
        
        # Create mapping of sid to code
        pid_mapping = {}
        for _, row in buggy_submissions.iterrows():
            sid = row['sid']
            pid = row['pid']
            pid_mapping[sid] = pid

        assert len(pid_mapping) == NUM_SIDS, \
            f"Expected {NUM_SIDS} PIDs, but found {len(pid_mapping)} in {dataset_path}"
        return pid_mapping
        
    except Exception as e:
        print_warning(f"Could not load PID mapping: {e}")
        return {}
    


def get_query_data_by_sid(sid: int, file_path: str = None) -> Optional[Dict[str, Any]]:
    """
    Retrieve complete query data from GAIED queries by SID.
    
    Args:
        sid (int): The SID (Student ID) to search for
        file_path (str, optional): Path to gaide_queries.json file
    
    Returns:
        Optional[Dict[str, Any]]: The complete query object if found, None otherwise
    
    Example:
        >>> data = get_query_data_by_sid(1)
        >>> if data:
        ...     print(f"SID: {data['sid']}, Prompt length: {len(data['prompt'])}")
    """
    queries = load_gaied_queries(file_path)
    
    for query in queries:
        if query.get('sid') == sid:
            return query
    
    print_warning(f"No query data found for SID {sid}")
    return None

def get_all_sids(file_path: str = None) -> List[int]:
    """
    Get all available SIDs from GAIED queries.
    
    Args:
        file_path (str, optional): Path to gaide_queries.json file
    
    Returns:
        List[int]: List of all SIDs found in the queries
    
    Example:
        >>> sids = get_all_sids()
        >>> print(f"Available SIDs: {len(sids)} total")
    """
    queries = []

    try:
        queries = load_gaied_queries(file_path)
        assert len(queries) == NUM_SIDS, "Number of SIDs does not match expected count"
    except Exception as e:
        print_warning(f"Could not load GAIED queries: {e}")

    return [query.get('sid') for query in queries if query.get('sid') is not None]

def clear_gaied_cache():
    """
    Clear the cached GAIED queries data.
    Useful for testing or when the file might have changed.
    """
    global _gaied_queries_cache
    _gaied_queries_cache = None
    print_warning("GAIED queries cache cleared")

def load_existing_results(model: str) -> List[Dict[str, Any]]:
    """
    Load existing results from a JSON file.
    
    Args:
        path_existing (str): Path to the existing results file
        
    Returns:
        List[Dict[str, Any]]: List of existing results, empty list if file doesn't exist or error occurs
    """
    path_existing = os.path.join('data', 'generator', f'{model}_feedback.json')

    existing_results = []
    if os.path.exists(path_existing):
        try:
            with open(path_existing, 'r') as f:
                existing_results = json.load(f)

            assert len(existing_results) == NUM_SIDS, "Number of existing results does not match expected count"
        except Exception as e:
            print_warning(f"Could not load existing results: {e}")
    
    return existing_results

def get_processed_sids(existing_results: List[Dict[str, Any]], category_required: bool) -> set:
    """
    Get set of already processed SIDs from existing results.
    Only includes SIDs with valid Pydantic-validated data structure.
    
    Args:
        existing_results (List[Dict[str, Any]]): List of existing results
        
    Returns:
        set: Set of SIDs that have already been processed with valid data
    """
    
    # Use Pydantic validation to determine which SIDs have valid data
    valid_sids = validate_json_file_data(existing_results, category_required=category_required)
    processed_sids = set(valid_sids)
    
    total_sids = len([result for result in existing_results if result.get('sid') is not None])
    assert total_sids == NUM_SIDS, f"Expected {NUM_SIDS} SIDs, but found {total_sids} in existing results"

    numInvalid = len(existing_results) - len(processed_sids)
    if numInvalid > 0:
        print_warning(f"{numInvalid} out of {len(existing_results)} existing results have invalid data")
    
    return processed_sids

def get_processed_results(model) -> GenerationBatch:
    """
    Filter out SIDs that have already been processed.
    
    Args:
        sids (List[int]): List of SIDs to filter
        processed_sids (set): Set of already processed SIDs
        
    Returns:
        List[int]: List of SIDs that have not been processed yet
    """
    category_required = model in MODELS_GEN
    
    existing_results = load_existing_results(model)
    if not existing_results:
        print_warning(f"No existing results found for model {model}. Generating new batch.")
        return GenerationBatch(generator_model=model, results=[], use_ground_truth=category_required)
    
    processed_sids = get_processed_sids(existing_results, category_required)
    return GenerationBatch(
        generator_model=model,
        results=[item for item in existing_results if item.get('sid') in processed_sids],
        use_ground_truth=category_required
    )



def validate_generator_data(data: Dict[str, Any], category_required: bool = True) -> bool:
    """
    Validate that data matches the complete Generator data structure.
    
    Args:
        data: Data dictionary to validate
        category_required: Whether category field is required in feedback
        
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        # Add category_required to data if not present
        if 'category_required' not in data:
            data['category_required'] = category_required

        # Handle line_number ranges like "1-3"
        if VALIDATOR_REPAIR.line_number_hyphens:
            if 'feedback' in data:
                for line in data['feedback']:
                    if 'line_number' in line and isinstance(line['line_number'], str):
                        if '-' in line['line_number']:
                            line['line_number'] = line['line_number'].split('-')[0]

        GeneratorData(**data)
        return True
    except ValidationError as e:
        print(f"- {e}")
        return False

def validate_json_file_data(data: List[Dict[str, Any]], category_required: bool = True) -> List[int]:
    """
    Validate data from a JSON file and return list of valid SIDs.
    
    Args:
        data: List of data entries from JSON file
        category_required: Whether category field is required in feedback
        
    Returns:
        List[int]: List of SIDs that have valid data structure
    """
    valid_sids = []
    student_code_mapping = load_student_code_mapping()
    
    for entry in data:
        try:
            # Add category_required to entry if not present
            if 'category_required' not in entry:
                entry['category_required'] = category_required
                
            # Validate the entry structure
            if validate_generator_data(entry, category_required):
                sid = entry['sid']
                
                # Check if student_code matches the dataset
                expected_student_code = student_code_mapping.get(sid)
                if expected_student_code is not None:
                    actual_student_code = entry.get('student_code', '')
                    
                    # Normalize whitespace for comparison
                    expected_normalized = ' '.join(expected_student_code.split())
                    actual_normalized = ' '.join(actual_student_code.split())
                    
                    if expected_normalized == actual_normalized:
                        valid_sids.append(sid)
                    else:
                        print_warning(f"Student code mismatch for sid {sid}")
                else:
                    print_warning(f"No expected student code found for sid {sid}")
            else:
                print_warning(f"SID {entry.get('sid', 'unknown')} validation failed")

        except Exception as e:
            print_error(f"Error validating entry: {e}")
            
    return valid_sids