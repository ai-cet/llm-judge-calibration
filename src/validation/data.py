"""
Data access utilities for validation.
"""

from collections import defaultdict
from typing import Tuple, Dict, Any, List, Optional
import json
import os
import sys
import re
from functools import lru_cache

from jsonschema import ValidationError

from src.config import MODELS_GEN, NUM_SIDS, VALIDATOR_REPAIR, pathValidator

from src.utils import print_warning, print_error
from src.validation.models import (
    GroundTruthFeedback, ValidationOutput, ValidationResult, ValidationBatch
)
from src.generation.models import GenerationBatch
import traceback
from src.generation.data import (load_existing_results, get_processed_results)


class DataProvider:
    """Centralized data provider for validation operations."""
    
    def __init__(self, modelGen: str, modelVal: str) -> None:
        self.validation_batch: Optional[ValidationBatch] = None
        self.generation_batch: Optional[GenerationBatch] = None
        self.error_message_counts = defaultdict(int)
        
        self.load_generation_batch(modelGen)
        self.load_validation_batch(modelGen, modelVal)
        
        # Print summary
        self.print_validation_summary()

        # Analyze error patterns
        error_analysis = self.analyze_error_patterns()
        if error_analysis['total_failed'] > 0:
            print(f"\nError Analysis:")
            print(f"Total Failed: {error_analysis['total_failed']}")
            for error_type, count in error_analysis['error_counts'].items():
                print(f"  {error_type}: {count} SIDs")

    def load_validation_batch(self, modelGen: str, modelVal: str) -> Optional[ValidationBatch]:
        """
        Load a validation batch from a JSON file.
        
        Args:
            modelGen: Model used for generation
            modelVal: Model used for validation

        Returns:
            ValidationBatch or None: Loaded batch if successful
        """
        try:
            file_path = f'{pathValidator}/gen={modelGen}/val={modelVal}.json'
            # file_path = f'{pathValidator}/new_labeller_gen_{modelGen}_val_{modelVal}_2025-05-05_13-06-52.json'
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Only keep sid=14 data
            # data = [item for item in data if item.get('sid') == 14]

            # If data is a list (old format), convert to new format
            if isinstance(data, list):
                # Try to extract metadata from filename
                filename = os.path.basename(file_path)
                use_ground_truth = "cheat" in filename

                # Convert list to ValidationResult objects
                results = []
                for item in data:
                    sid = item.get('sid', -1)
                    generatorData = self.generation_batch.getById(sid) if self.generation_batch else None
                    item['generatorData'] = generatorData

                    # Handle issues in the output structure
                    if 'output' in item and isinstance(item['output'], dict) and 'feedback_lines' in item['output']:
                        for feedback_line in item['output']['feedback_lines']:
                            # Ensure 'line_num' is converted to 'line_number'
                            if VALIDATOR_REPAIR.line_num_number:
                                if isinstance(feedback_line, dict) and 'line_num' in feedback_line and 'line_number' not in feedback_line:
                                    feedback_line['line_number'] = feedback_line.pop('line_num')

                            # Handle line_number ranges like "1-3"
                            if VALIDATOR_REPAIR.line_number_hyphens:
                                if 'line_number' in feedback_line and isinstance(feedback_line['line_number'], str):
                                    if '-' in feedback_line['line_number']:
                                        feedback_line['line_number'] = feedback_line['line_number'].split('-')[0]

                            # If the feedback line has a classification and it's 'partially valid', change it to 'invalid'
                            if VALIDATOR_REPAIR.partially_valid_label: 
                                if 'classification' in feedback_line and feedback_line['classification'] == 'partially valid':
                                    feedback_line['classification'] = 'invalid'

                    try:
                        # Ensure item has required fields and attach it
                        result = ValidationResult(**item)
                        results.append(result)
                        self.error_message_counts['unmatched_feedback'] += result.get_countFids_failure()

                    except Exception as e:
                        # Handle parsing errors gracefully
                        error_lines = str(e).splitlines()
                        error_lines_alt = [error_lines[i] for i in range(1, len(error_lines), 3)]
                        error_str = " | ".join(error_lines_alt)
                        fidFailureCount = 0 # len(generatorData.feedback) if generatorData else 0
                        print_warning(f"Error parsing result for SID {item.get('sid', 'unknown')}: {e}")

                        # Maintain a count of error messages
                        error_str = str(e)
                        match = re.search(r"\[ID=(.*?)\]", error_str)
                        error_message = match.group(1) if match else error_str

                        # If missing output
                        if error_message.startswith('1 validation error for ValidationResult\noutput\n'):
                            error_message = 'missing_output'

                        if re.search(r"output\.feedback_lines\.\d+\.classification", error_str):
                            error_message = 'missing_label'

                        # For each generator line, increment the error message count
                        if 'generatorData' in item and item['generatorData'] is not None:
                            numGenLines = len(item['generatorData'].feedback)
                            if numGenLines > 0:
                                self.error_message_counts[error_message] += numGenLines

                        # Try to atleast create a minimal result
                        minimal_result = ValidationResult(
                            generatorData=None,
                            sid=item.get('sid', -1),
                            output=None,
                            fidFailureCount=fidFailureCount,
                            raw_response=item.get('raw_response', ''),
                            error=error_str
                        )
                        results.append(minimal_result)

                self.validation_batch = ValidationBatch(
                    generator_model=modelGen,
                    validator_model=modelVal,
                    results=results,
                    use_ground_truth=use_ground_truth
                )
            else:
                # New format
                self.validation_batch = ValidationBatch(**data)

        except Exception as e:
            print_error(f"Error loading validation batch for gen={modelGen}, val={modelVal}: {e}")
            print_warning(f"Full traceback: {traceback.format_exc()}")
            
            sys.exit(1)
            self.validation_batch = None

    def save_validation_batch_to_file(self, file_path: str) -> bool:
        """
        Save a validation batch to a JSON file.
        
        Args:
            file_path: Path to save the batch

        Returns:
            bool: True if successful, False otherwise
        """
        if self.validation_batch is None:
            print_warning("No validation batch to save.")
            return False

        try:
            with open(file_path, 'w') as f:
                json.dump(self.validation_batch.dict(), f, indent=4)

            return True

        except Exception as e:
            print_error(f"Error saving validation batch to {file_path}: {e}")
            return False

    def get_total_fids_count(self) -> int:
        """
        Get all FID results from the validation batch.

        Returns:
            int: Count of all FID results
        """
        if self.validation_batch is None:
            return 0

        return sum(
            result.get_countFids_failure() + result.get_countFids_success()
            for result in self.validation_batch.results
        )

    def get_failed_sids(self) -> List[int]:
        """
        Get list of SIDs that failed validation.
        
        Returns:
            List[int]: List of failed SIDs
        """
        if self.validation_batch is None:
            return []

        return [result.sid for result in self.validation_batch.results if result.is_failed_sid()]
    
    def get_failed_sids_partial(self) -> List[int]:
        """
        Get list of SIDs that failed partial validation.

        Returns:
            List[int]: List of failed SIDs
        """
        if self.validation_batch is None:
            return []

        return [result.sid for result in self.validation_batch.results if result.is_failed_fid()]

    def get_failed_fids_count(self) -> int:
        """
        Get the count of failed FIDs.
        
        Returns:
            int: Count of failed FIDs
        """
        if self.validation_batch is None:
            return 0

        return sum(result.fidFailureCount for result in self.validation_batch.results)

    def print_validation_summary(self) -> None:
        """
        Print a comprehensive summary of validation results.
        """
        batch = self.validation_batch
        if batch is None:
            print_warning("No validation batch loaded.")
            return
        stats = batch.get_summary_stats()
        
        print(f"\n{'='*60}")
        print(f"VALIDATION SUMMARY: {batch.generator_model} -> {batch.validator_model}")
        print(f"{'='*60}")
        print(f"Generator Model: {batch.generator_model}")
        print(f"Validator Model: {batch.validator_model}")
        print(f"")
        print(f"Successful SIDs: {stats['successful_results']} / {stats['total_results']} ({stats['success_rate']:.2%})")
        print(f"Successful FIDs: {stats['successful_fids']} / {stats['total_fids']} ({stats['successful_fids'] / stats['total_fids']:.2%})")
        print(f"")
        print(f"Classification Results:")
        print(f"  Valid Feedback: {stats['total_valid']}")
        print(f"  Invalid Feedback: {stats['total_invalid']}")
        print(f"  Precision: {stats['precision']:.4f}")
        print(f"{'='*60}")
        
        # Print failed SIDs if any
        failed_sids = self.get_failed_sids()
        if failed_sids:
            print(f"\nSIDs with complete failures ({len(failed_sids)}): {failed_sids}")

        failed_fids = self.get_failed_sids_partial()
        if failed_fids:
            print(f"SIDs with partial failures ({len(failed_fids)}): {failed_fids}")

    def analyze_error_patterns(self) -> Dict[str, Any]:
        """
        Analyze error patterns in validation results.
        
        Returns:
            Dict containing error analysis
        """
        failed_results = [r for r in self.validation_batch.results if r.is_failed_sid()]
        
        error_patterns = {}
        for result in failed_results:
            # Extract error type from raw response
            error_key = "unknown_error"
            
            if "429" in result.raw_response:
                error_key = "quota_exceeded"
            elif "timeout" in result.raw_response.lower():
                error_key = "timeout"
            elif "connection" in result.raw_response.lower():
                error_key = "connection_error"
            elif "json" in result.raw_response.lower():
                error_key = "json_parse_error"
            elif "error" in result.raw_response.lower():
                error_key = "general_error"
            
            if error_key not in error_patterns:
                error_patterns[error_key] = []
            error_patterns[error_key].append(result.sid)
        
        return {
            "total_failed": len(failed_results),
            "error_patterns": error_patterns,
            "error_counts": {k: len(v) for k, v in error_patterns.items()}
        }
    
    def load_generation_batch(self, modelGen: str):
        print(f"\n{'='*60}")
        print(f"GENERATION SUMMARY: {modelGen}")
        print(f"{'='*60}")

        processed_results = get_processed_results(modelGen)
        self.generation_batch = processed_results



def convert_ground_truth_categories(ground_truth: List[Dict[str, Any]]) -> List[GroundTruthFeedback]:
    """
    Convert ground truth categories from detailed format to valid/invalid.
    
    Args:
        ground_truth: List of ground truth feedback with detailed categories
        
    Returns:
        List[GroundTruthFeedback]: Converted ground truth feedback
    """
    converted_feedback = []
    
    for item in ground_truth:
        # Convert detailed categories to valid/invalid
        category = item.get('category', '')
        if category in ['TP', 'FP-E', 'FP-R', 'TP-E', 'TP-R']:
            new_category = 'valid'
        elif category in ['FP-H', 'FP-I']:
            new_category = 'invalid'
        else:
            new_category = category  # Keep as is if already valid/invalid
        
        converted_item = {
            'line_number': item['line_number'],
            'feedback': item['feedback'],
            'category': new_category
        }
        
        try:
            converted_feedback.append(GroundTruthFeedback(**converted_item))
        except ValidationError as e:
            print_warning(f"Error converting ground truth item: {e}")
    
    return converted_feedback


def validate_json_file_data(file_path: str) -> List[ValidationResult]:
    """
    Validate and parse validation results from a JSON file.
    
    Args:
        file_path: Path to the JSON file containing validation results
        
    Returns:
        List[ValidationResult]: List of valid validation results
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            print_error(f"Expected list in JSON file {file_path}")
            return []
        
        valid_results = []
        for item in data:
            try:
                # Parse the output if it exists and is not empty
                output = None
                if item.get('output') and isinstance(item['output'], dict):
                    try:
                        output = ValidationOutput(**item['output'])
                    except ValidationError as e:
                        print_warning(f"Invalid output format for SID {item.get('sid', 'unknown')}: {e}")
                
                result = ValidationResult(
                    sid=item['sid'],
                    raw_response=item['raw_response'],
                    output=output,
                    success=output is not None,
                    generator_model=item.get('generator_model', 'unknown'),
                    validator_model=item.get('validator_model', 'unknown'),
                    timestamp=item.get('timestamp')
                )
                valid_results.append(result)
                
            except ValidationError as e:
                print_warning(f"Error validating result for SID {item.get('sid', 'unknown')}: {e}")
        
        return valid_results
        
    except Exception as e:
        print_error(f"Error reading validation file {file_path}: {e}")
        return []
