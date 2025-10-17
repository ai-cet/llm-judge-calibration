"""
Feedback generation script using the unified model invocation system.
Generates feedback for student code using various AI models with retry logic and structured JSON parsing.
"""

import json
import os
import traceback
from tqdm import tqdm

from src.config import MODELS_GEN, Model

from src.LLM import invoke_model_with_retry, GenerationOutput
from src.generation.utils import (
    generator_data_failure,
    transform_llm_output_to_generator_data,
    validate_and_save_generator_data
)
from src.generation.models import validate_generator_data, validate_llm_output, GeneratorData
from src.generation.data import (load_student_code_mapping, load_pid_mapping, get_all_sids, get_query_data_by_sid, 
    load_existing_results, get_processed_sids)
from src.utils import print_warning, print_error

####################
# Config
####################

### Missing annotations
# CLAUDE_3_OPUS: 2 SIDs
# GPT_4_TURBO: 17 SIDs
# GPT_4O: 2 SIDs

# Current selected model
model = Model.CLAUDE_3_5_HAIKU.value

# Configuration for category requirement
# Set to False if generating initial feedback without categories
# Set to True if categories are required (for labeled data)
category_required = True if model in MODELS_GEN else False


####################
# Init vars
####################

results = []
available_sids = []
sids = []
student_code_mapping = None

####################
# Setup vars
####################
path_existing = f'./data/generator/{model}_feedback.json'

####################
# Read existing data
####################

if model is None:
    raise ValueError("Please select a model by uncommenting one of the model lines above")

print(f"Using model: {model}")
print(f"Output path: {path_existing}")

# Load available SIDs from GAIED queries
available_sids = get_all_sids()

# Load existing results if available
existing_results = load_existing_results(model)

# Get list of already processed SIDs (only those with valid Pydantic structure)
processed_sids = get_processed_sids(existing_results, category_required=category_required)

# Load student code mapping for data transformation
student_code_mapping = load_student_code_mapping()

# Load PID to SID mapping if available
pid_mapping = load_pid_mapping()

####################
# Generation function
####################

def generate_feedback_for_sid(sid: int) -> 'GeneratorData':
    """
    Generate feedback for a single SID using the unified model system.
    Now returns validated Generator data format.
    
    Args:
        sid (int): The SID to process
        
    Returns:
        GeneratorData: Result with SID, generator data, and success status
    """
    
    # Get query data for this SID
    sid_data = get_query_data_by_sid(sid)
    if not sid_data:
        return generator_data_failure(sid, f'No query data found for SID {sid}')

    prompt = sid_data.get('prompt', '')

    if not prompt:
        return generator_data_failure(sid, 'No prompt found for this SID')

    try:
        # Create messages for the model
        messages = [
            {
                "role": "user", 
                "content": prompt
            }
        ]
        
        print(f"Processing SID {sid}...")
        
        # Use unified model invocation with retry logic
        raw_response, parsed_response, success = invoke_model_with_retry(
            provider=None,  # Auto-detect provider from model name
            model=model,
            messages=messages,
            expected_output_model=GenerationOutput,
            max_retries=3
        )
        
        if not success or not parsed_response:
            return generator_data_failure(sid, 'Failed to generate valid JSON response after retries')

        # Validate LLM response format
        if not validate_llm_output(parsed_response):
            return generator_data_failure(sid, 'LLM response does not match expected format')

        pid = pid_mapping.get(sid, sid)  # Use SID as PID if no mapping available
        # Transform to Generator data format
        generator_data = transform_llm_output_to_generator_data(
            llm_output=parsed_response,
            sid=sid,
            pid=pid,
            category_required=category_required
        )
        
        if generator_data is None:
            return generator_data_failure(sid, 'Failed to transform LLM output to Generator data format')

        result = GeneratorData(
            sid=sid,
            generator_data=generator_data,
            success=True,
            model=model
        )
        
        print(f"✅ Successfully processed SID {sid}")
        return result
        
    except Exception as e:
        error_msg = f"Exception occurred: {str(e)}"
        print_error(f"Error processing SID {sid}: {error_msg}")
        traceback.print_exc()

        return generator_data_failure(sid, error_msg)

####################
# Main processing loop
####################

def main():
    """Main processing function."""
    global results, existing_results
    
    # Filter out already processed SIDs
    remaining_sids = [sid for sid in available_sids if sid not in processed_sids]
    
    if not remaining_sids:
        print("No new SIDs to process!")
        return
    
    print(f"Processing {len(remaining_sids)} remaining SIDs...")
    
    # Start with existing results that have valid Generator data structure
    results = []
    
    # Re-validate existing results and only keep valid ones
    for existing_result in existing_results:
        if existing_result.get('success') and existing_result.get('generator_data'):
            # Validate the generator data structure

            if validate_generator_data(existing_result['generator_data'], category_required):
                results.append(existing_result)
            else:
                print_warning(f"Existing result for SID {existing_result.get('sid')} failed validation")
    
    print(f"Kept {len(results)} valid existing results")
    
    # Process each SID with progress bar
    for sid in tqdm(remaining_sids, desc=f"Generating feedback with {model}"):
        try:
            result = generate_feedback_for_sid(sid)
            results.append(result)
            
            # If successful, also save the generator data to a separate validated file
            if result.get('success') and result.get('generator_data'):
                # Save individual validated generator data
                validated_path = path_existing.replace('.json', '_validated.json')
                if not validate_and_save_generator_data(result['generator_data'], validated_path, category_required):
                    print_warning(f"Failed to save validated data for SID {sid}")
            
            # Save progress periodically (every 10 items)
            if len(results) % 10 == 0:
                save_results()
                
        except KeyboardInterrupt:
            print_warning("\nProcess interrupted by user")
            save_results()
            break
        except Exception as e:
            print_error(f"Unexpected error: {e}")
            traceback.print_exc()
            continue
    
    # Final save
    save_results()
    
    # Print summary
    print_summary()

def save_results():
    """Save current results to file."""
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(path_existing), exist_ok=True)
        
        # Save all results (including failed ones for debugging)
        with open(path_existing, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Saved {len(results)} results to {path_existing}")
        
        # Also save only the validated generator data
        validated_results = []
        for result in results:
            if result.get('success') and result.get('generator_data'):
                validated_results.append(result['generator_data'])
        
        if validated_results:
            validated_path = path_existing.replace('.json', '_generator_data.json')
            with open(validated_path, 'w') as f:
                json.dump(validated_results, f, indent=2)
            print(f"💾 Saved {len(validated_results)} validated generator data entries to {validated_path}")
            
    except Exception as e:
        print_error(f"Error saving results: {e}")

def print_summary():
    """Print processing summary."""
    total = len(results)
    successful = sum(1 for r in results if r.get('success', False))
    failed = total - successful
    validated = sum(1 for r in results if r.get('success') and r.get('generator_data'))
    
    print("\n" + "="*60)
    print("GENERATION SUMMARY")
    print("="*60)
    print(f"Model: {model}")
    print(f"Total processed: {total}")
    print(f"Successful: {successful} ({successful/total*100:.1f}%)")
    print(f"Failed: {failed} ({failed/total*100:.1f}%)")
    print(f"Validated Generator Data: {validated} ({validated/total*100:.1f}%)")
    print(f"Output file: {path_existing}")
    print(f"Validated data file: {path_existing.replace('.json', '_generator_data.json')}")
    print("="*60)

if __name__ == "__main__":
    try:
        pass
        # main()
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        traceback.print_exc()
        # Try to save whatever we have
        if results:
            save_results()