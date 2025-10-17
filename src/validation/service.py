"""
Service layer for validation operations.
"""

from typing import Tuple, Dict, Any, Optional, List
import datetime
import json

from src.validation.models import (
    ValidationOutput, ValidationResult
)
from src.validation.data import DataProvider
from src.validation.prompt import get_validation_prompt, get_validation_prompt_with_ground_truth
from src.LLM import invoke_model_with_retry, ValidationOutput as LLMValidationOutput
from src.utils import print_warning, print_error


class ValidationService:
    """Service for handling validation operations."""
    
    def __init__(self, generator_model: str, validator_model: str, use_ground_truth: bool = False):
        """
        Initialize the validation service.
        
        Args:
            generator_model: Model used for generation
            validator_model: Model used for validation
            use_ground_truth: Whether to use ground truth data
        """
        self.generator_model = generator_model
        self.validator_model = validator_model
        self.use_ground_truth = use_ground_truth
        self.data_provider = DataProvider()
    
    def validate_single_sid(self, sid: int) -> ValidationResult:
        """
        Validate feedback for a single SID.
        
        Args:
            sid: Student ID to validate
            
        Returns:
            ValidationResult: Result of the validation
        """
        try:
            # Get validation input data
            validation_input = self.data_provider.create_validation_input(
                sid, self.generator_model, self.use_ground_truth
            )
            
            # Generate prompt messages
            if self.use_ground_truth and validation_input.ground_truth:
                messages = get_validation_prompt_with_ground_truth(
                    validation_input.question,
                    validation_input.student_code,
                    validation_input.correct_code,
                    [fb.dict() for fb in validation_input.feedback],
                    [tc.dict() for tc in validation_input.test_cases],
                    [gt.dict() for gt in validation_input.ground_truth]
                )
            else:
                messages = get_validation_prompt(
                    validation_input.question,
                    validation_input.student_code,
                    validation_input.correct_code,
                    [fb.dict() for fb in validation_input.feedback],
                    [tc.dict() for tc in validation_input.test_cases]
                )
            
            # TODO: Don't invoke LLMs unless specified in a class variable. Also, invoke LLM only after validation output failure (on reading the file)
            # TODO: After invoking LLM, store each intermediate result along with number of tries, to understand the mistakes made by LLM in each try. Eg. "iteration": 2, "issue": "missing-line-number", "fix": "rename line-num to line-number"
            # Invoke model with retry logic
            raw_response, parsed_response, success = invoke_model_with_retry(
                provider=None,  # Auto-detect provider
                model=self.validator_model,
                messages=messages,
                expected_output_model=LLMValidationOutput
            )
            
            # Convert parsed response to ValidationOutput if successful
            validation_output = None
            if success and parsed_response:
                try:
                    validation_output = ValidationOutput(**parsed_response)
                except Exception as e:
                    print_warning(f"Error parsing validation output for SID {sid}: {e}")
                    success = False
            
            # Create result
            return ValidationResult(
                sid=sid,
                raw_response=str(raw_response),
                output=validation_output,
                success=success,
                generator_model=self.generator_model,
                validator_model=self.validator_model,
                timestamp=datetime.datetime.now().isoformat()
            )
            
        except Exception as e:
            print_error(f"Error validating SID {sid}: {e}")
            return ValidationResult(
                sid=sid,
                raw_response=f"Error: {str(e)}",
                output=None,
                success=False,
                generator_model=self.generator_model,
                validator_model=self.validator_model,
                timestamp=datetime.datetime.now().isoformat()
            )
    
    def validate_batch(self, sids: List[int]) -> List[ValidationResult]:
        """
        Validate feedback for a batch of SIDs.
        
        Args:
            sids: List of Student IDs to validate
            
        Returns:
            List[ValidationResult]: Results of the validations
        """
        results = []
        
        for sid in sids:
            print(f"Validating SID {sid}...")
            result = self.validate_single_sid(sid)
            results.append(result)
            
            # Print progress
            if result.success and result.output:
                counts = result.get_classification_counts()
                print(f"SID {sid}: Valid={counts['valid']}, Invalid={counts['invalid']}")
            else:
                print(f"SID {sid}: Failed")
        
        return results


def create_attempt_name(generator_model: str, validator_model: str, use_ground_truth: bool = False) -> str:
    """
    Create a standardized attempt name for validation runs.
    
    Args:
        generator_model: Model used for generation
        validator_model: Model used for validation
        use_ground_truth: Whether ground truth was used
        
    Returns:
        str: Formatted attempt name
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    if use_ground_truth:
        return f"new_labeller_gen_{generator_model}_val_{validator_model}_cheat_{timestamp}"
    else:
        return f"new_labeller_gen_{generator_model}_val_{validator_model}_{timestamp}"


def calculate_precision_stats(results: List[ValidationResult]) -> Dict[str, Any]:
    """
    Calculate precision statistics from validation results.
    
    Args:
        results: List of validation results
        
    Returns:
        Dict containing precision statistics
    """
    total_valid = 0
    total_invalid = 0
    successful_results = 0
    
    for result in results:
        if result.success and result.output:
            successful_results += 1
            counts = result.get_classification_counts()
            total_valid += counts['valid']
            total_invalid += counts['invalid']
    
    total_feedback = total_valid + total_invalid
    precision = total_valid / total_feedback if total_feedback > 0 else 0
    
    return {
        'total_results': len(results),
        'successful_results': successful_results,
        'success_rate': successful_results / len(results) if results else 0,
        'total_valid': total_valid,
        'total_invalid': total_invalid,
        'total_feedback': total_feedback,
        'precision': precision
    }


def print_validation_stats(results: List[ValidationResult], prefix: str = "") -> None:
    """
    Print validation statistics in a formatted way.
    
    Args:
        results: List of validation results
        prefix: Optional prefix for the output
    """
    stats = calculate_precision_stats(results)
    
    print(f"{prefix}Validation Statistics:")
    print(f"  Total Results: {stats['total_results']}")
    print(f"  Successful: {stats['successful_results']} ({stats['success_rate']:.2%})")
    print(f"  Total Valid: {stats['total_valid']}")
    print(f"  Total Invalid: {stats['total_invalid']}")
    print(f"  Precision: {stats['precision']:.4f}")


class ValidationRunner:
    """High-level runner for validation operations."""
    
    def __init__(self, generator_model: str, validator_model: str, use_ground_truth: bool = False):
        """
        Initialize the validation runner.
        
        Args:
            generator_model: Model used for generation
            validator_model: Model used for validation
            use_ground_truth: Whether to use ground truth data
        """
        self.service = ValidationService(generator_model, validator_model, use_ground_truth)
        self.attempt_name = create_attempt_name(generator_model, validator_model, use_ground_truth)
        self.results_file = f"./new_logs/{self.attempt_name}.json"
    
    def run_validation(self, sids: List[int], resume_from_file: Optional[str] = None) -> None:
        """
        Run validation for a list of SIDs with progress tracking and file saving.
        
        Args:
            sids: List of Student IDs to validate
            resume_from_file: Optional file to resume from (load existing results)
        """
        from src.validation.data import load_existing_validation_results, update_validation_result_in_file
        
        # Load existing results if resuming
        if resume_from_file:
            existing_results = load_existing_validation_results(resume_from_file)
            processed_sids = {r['sid'] for r in existing_results}
            sids = [sid for sid in sids if sid not in processed_sids]
            print(f"Resuming validation. Skipping {len(processed_sids)} already processed SIDs.")
            print(f"Remaining SIDs to process: {len(sids)}")
        
        # Track cumulative statistics
        cumulative_valid = 0
        cumulative_invalid = 0
        
        for i, sid in enumerate(sids, 1):
            print(f"\n[{i}/{len(sids)}] Processing SID {sid}...")
            
            # Validate single SID
            result = self.service.validate_single_sid(sid)
            
            # Update cumulative stats
            if result.success and result.output:
                counts = result.get_classification_counts()
                cumulative_valid += counts['valid']
                cumulative_invalid += counts['invalid']
                
                # Calculate current precision
                total_feedback = cumulative_valid + cumulative_invalid
                precision = cumulative_valid / total_feedback if total_feedback > 0 else 0
                
                print(f"SID {sid}: Valid={counts['valid']}, Invalid={counts['invalid']}")
                print(f"Cumulative - Precision: {precision:.4f}, Valid: {cumulative_valid}, Invalid: {cumulative_invalid}")
            else:
                print(f"SID {sid}: Failed to get valid response")
            
            # Save result to file
            result_dict = result.dict()
            if update_validation_result_in_file(self.results_file, sid, result_dict):
                print(f"Saved result for SID {sid}")
            else:
                print_error(f"Failed to save result for SID {sid}")
        
        print(f"\nValidation completed. Results saved to {self.results_file}")
    
    def run_quota_recovery(self, original_file: str) -> None:
        """
        Run validation for SIDs that failed due to quota exceeded errors.
        
        Args:
            original_file: Path to the original results file with quota errors
        """
        from src.validation.data import filter_quota_exceeded_sids
        
        # TODO: Filter all erroneous SIDs from the original file
        failed_sids = filter_quota_exceeded_sids(original_file)
        
        if not failed_sids:
            print("No SIDs found with quota exceeded errors.")
            return
        
        print(f"Found {len(failed_sids)} SIDs with quota exceeded errors: {failed_sids}")
        
        # Run validation for failed SIDs
        self.run_validation(failed_sids, resume_from_file=original_file)
