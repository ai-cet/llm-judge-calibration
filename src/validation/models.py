"""
Pydantic models for Validator data validation.
"""

import sys
from typing import Annotated, List, Dict, Any, Optional, Union
import pandas as pd
from pydantic import BaseModel, BeforeValidator, ValidationError, validator, Field, model_validator
from src.config import VALIDATOR_REPAIR
from src.generation.models import GeneratorData
from src.utils import print_warning, print_error
import json
from fuzzywuzzy import fuzz
from collections import OrderedDict


####################
# Pydantic Models
####################

class GroundTruthFeedback(BaseModel):
    """Model for ground truth feedback with category."""
    line_number: Union[str, int] = Field(..., description="Line number where the feedback applies")
    feedback: str = Field(..., description="Feedback text for the student")
    category: str = Field(..., description="Ground truth category (valid/invalid)")
    
    @validator('line_number', pre=True)
    def validate_line_number(cls, v):
        """Convert line_number to string for consistency."""
        return str(v)


class ValidatedFeedbackLine(BaseModel):
    """Model for feedback line with validation results."""
    line_number: Union[str, int] = Field(..., description="Line number referenced by the feedback")
    feedback: str = Field(..., description="The feedback provided by the TA")
    analysis: str = Field(..., description="Analysis of the feedback's accuracy")
    classification: str = Field(..., description="Feedback validity classification (valid/invalid)")
    
    @validator('line_number', pre=True)
    def validate_line_number(cls, v):
        """Convert line_number to string for consistency."""
        return str(v)
    
    @validator('classification')
    def validate_classification(cls, v):
        """Ensure classification is either 'valid' or 'invalid'."""
        if VALIDATOR_REPAIR.partially_valid_label:
            # If partially valid is allowed, treat it as invalid for consistency
            if v.lower() == 'partially valid':
                return 'invalid'
            
        if v.lower() not in ['valid', 'invalid']:
            raise ValueError("[ID=invalid_label] Classification must be either 'valid' or 'invalid'")
        return v.lower()


class ValidationOutput(BaseModel):
    """Model for LLM validation output format."""
    mistakes: Optional[List[str]] = Field([], description="List of mistakes found in the student's code")
    fixes: Optional[List[str]] = Field([], description="List of corrections proposed in the fixed code")
    feedback_lines: List[ValidatedFeedbackLine] = Field(..., description="List of validated feedback lines")

# Contains one validation result for a given submission ID (sid)
class ValidationResult(BaseModel):
    """Model for complete validation result."""
    generatorData: Optional[GeneratorData] = Field(None, description="Corresponding Generator data used for validation")
    sid: int = Field(..., description="Student ID")
    raw_response: str = Field(..., description="Raw LLM response")
    output: Optional[ValidationOutput] = Field(None, description="Parsed validation output")
    fidFailureCount: int = Field(0, description="Count of failed feedback lines")
    timestamp: Optional[str] = Field(None, description="Timestamp of validation")

            
    @classmethod
    def _get_feedback_items(cls, generator_data: GeneratorData, v: dict[str, Any]) -> List[tuple[str, str]]:
        """Extract feedback items from generator data."""
        if 'feedback_lines' not in v or not v['feedback_lines']:
            raise ValueError("[ID=missing_output] Validation output must contain 'feedback_lines'")

        # Create sets of (line_number, feedback) tuples for comparison
        generator_feedback_items = OrderedDict()
        for fb in generator_data.feedback:
            if not fb.line_number:
                raise ValueError("[ID=missing_generator_line_number] Generator feedback must contain 'line_number'")
            elif not fb.feedback:
                raise ValueError("[ID=missing_generator_feedback] Generator feedback must contain 'feedback'")

            generator_feedback_items[(str(fb.line_number), fb.feedback)] = None

        validation_feedback_items = OrderedDict()
        for fb_line in v['feedback_lines']:
            if 'line_number' not in fb_line:
                raise ValueError("[ID=missing_line_number] Validation feedback lines must contain 'line_number'")
            if 'feedback' not in fb_line:
                raise ValueError("[ID=missing_feedback] Validation feedback lines must contain 'feedback'")

            validation_feedback_items[(str(fb_line['line_number']), fb_line['feedback'])] = None

        return generator_feedback_items, validation_feedback_items

    @classmethod
    def _exact_match(cls, generator_data: dict[str, Any], output: dict[str, Any]) -> bool:
        generator_feedback_items, validation_feedback_items = cls._get_feedback_items(generator_data, output)
        generator_feedback_items = set(generator_feedback_items.keys())
        validation_feedback_items = set(validation_feedback_items.keys())

        # Check if all generator feedback items are present in validation output
        matched_items = generator_feedback_items & validation_feedback_items
        missing_items = generator_feedback_items - validation_feedback_items

        return matched_items, missing_items

    @classmethod
    def _fuzzy_match(cls, generator_data: dict[str, Any], output: dict[str, Any]) -> bool:
        # Create lists of (line_number, feedback) tuples for comparison
        generator_feedback_items, validation_feedback_items = cls._get_feedback_items(generator_data, output)
        
        # Sort feedback items to ensure deterministic processing
        # generator_feedback_items = sorted(generator_feedback_items)
        # validation_feedback_items = sorted(validation_feedback_items)

        matched_items = set()
        missing_items = set()
        # Track which validation items have been matched to avoid duplicates
        matched_validation_indices = set()
        # if generator_data.sid == 13:
        #     print(output['feedback_lines'])
        
        for gen_line, gen_feedback in generator_feedback_items:
            # Find best match for this generator feedback
            best_match_score = 0
            best_match_index = -1
            # print_error(f"Generator feedback: {gen_line}. {gen_feedback}")

            for val_index, (val_line, val_feedback) in enumerate(validation_feedback_items):
                if val_index in matched_validation_indices:
                    continue  # Skip already matched items

                if val_line != gen_line:  # Different line number
                    continue  # Skip if line numbers don't match

                # Clip validation feedback if it's shorter than generator feedback. To handle cases where validator got "lazy"
                gen_feedback_cmp = gen_feedback
                if VALIDATOR_REPAIR.clip_feedback_lazy:
                    if len(val_feedback) < len(gen_feedback):
                        gen_feedback_cmp = gen_feedback[:len(val_feedback)]

                if gen_line == val_line:  # Same line number
                    similarity = fuzz.ratio(gen_feedback_cmp, val_feedback)
                    if similarity > best_match_score:
                        best_match_score = similarity
                        best_match_index = val_index

                # if generator_data.sid == 13:
                #     print_error(f"\tGeneration feedback: {gen_line}. {gen_feedback_cmp}")
                #     print_error(f"\tValidation feedback: {val_line}. {val_feedback}")
                #     print_error(f"\tSimilarity score: {similarity}, Best match score: {best_match_score}, Best match index: {best_match_index}")
            
            # If good match found, replace validation feedback with generator feedback
            if best_match_score >= 85:  # 85% similarity threshold
                matched_validation_indices.add(best_match_index)
                # Replace the feedback text in the validation feedback line
                output['feedback_lines'][best_match_index]['feedback'] = gen_feedback
                matched_items.add((gen_line, gen_feedback))
            else:
                # If no good match found, mark as missing
                missing_items.add((gen_line, gen_feedback))

        return matched_items, missing_items

    @classmethod
    def _print_validate_output(cls, value: dict[str, Any], generator_data, matched_items, missing_items) -> None:
        """Print the validation output for debugging."""
        # Ensure we have at least one matched item
        if not matched_items:
            raise ValueError("[ID=unmatched_feedback] No matching feedback lines found in validation output")

        # If missing items, log a warning and add them as unsuccessful
        if missing_items:
            value['fidFailureCount'] = len(missing_items)  # Count of missing feedback items
            missing_str = "; ".join([f"Line {ln}: {fb}" for ln, fb in missing_items])
            print_warning(f"Missing feedback items for SID {value.get('sid', 'unknown')}: {missing_str}")

    @model_validator(mode='before')
    @classmethod
    def validate_output(cls, value: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the output field."""
        # Ensure output is present and has feedback_lines
        if value is None or 'output' not in value or not value['output']:
            return value
        
        # Ensure output is a dict
        if not isinstance(value['output'], dict):
            raise ValueError("[ID=missing_output] Validation output must be a dictionary")
        
        output = value['output']
        generator_data = value.get('generatorData')
        if not generator_data or not generator_data.feedback:
            return value
        
        # Check if output is a dict
        if 'feedback_lines' not in output or not output['feedback_lines']:
            raise ValueError("[ID=missing_output] Validation output must contain 'output' with 'feedback_lines'")

        # Choose matching strategy
        if VALIDATOR_REPAIR.feedback_match_fuzzy:
            matched_items, missing_items = cls._fuzzy_match(generator_data, output)
        else:
            matched_items, missing_items = cls._exact_match(generator_data, output)

        cls._print_validate_output(value, generator_data, matched_items, missing_items)

        return value
    
    def get_classification_counts(self) -> Dict[str, int]:
        """Get counts of valid/invalid classifications."""
        if self.is_failed_sid():
            return {"valid": 0, "invalid": 0}
        
        counts = {"valid": 0, "invalid": 0}
        for feedback_line in self.output.feedback_lines:
            classification = feedback_line.classification.lower()
            if classification in counts:
                counts[classification] += 1
        
        return counts   

    def get_countFids_failure(self) -> int:
        """Get the count of failed to validate feedback lines."""
        return self.fidFailureCount if self.fidFailureCount is not None else 0
    
    def get_countFids_success(self) -> int:
        """Get the total count of feedback lines."""
        if self.is_failed_sid():
            return 0
    
        return len(self.output.feedback_lines)
    
    def get_countFids_total(self) -> int:
        """Get the total count of feedback lines."""
        return self.get_countFids_success() + self.get_countFids_failure()
    
    def is_failed_sid(self) -> bool:
        """Check if the validation failed for all FIDs for this SID."""
        return not self.output or not self.output.feedback_lines

    def is_failed_fid(self) -> bool:
        """Check if any of the fid validation failed."""
        return self.is_failed_sid() or self.get_countFids_failure() > 0

# Contains the list of validator results for a given validator and generator model
class ValidationBatch(BaseModel):
    """Model for batch validation results."""
    generator_model: str = Field(..., description="Model used for generation")
    validator_model: str = Field(..., description="Model used for validation")
    results: List[ValidationResult] = Field(..., description="List of validation results")
    use_ground_truth: bool = Field(default=False, description="Whether ground truth was used")
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for the batch."""
        total_results = len(self.results)
        failed_results = sum([1 for r in self.results if r.is_failed_sid()])
        successful_results = total_results - failed_results

        failure_fids = sum([r.get_countFids_failure() for r in self.results])
        successful_fids = sum([r.get_countFids_success() for r in self.results])
        total_fids = successful_fids + failure_fids

        total_valid = 0
        total_invalid = 0
        
        for result in self.results:
            counts = result.get_classification_counts()
            total_valid += counts["valid"]
            total_invalid += counts["invalid"]
        
        precision = total_valid / (total_valid + total_invalid) if (total_valid + total_invalid) > 0 else 0
        
        return {
            "total_results": total_results,
            "successful_results": successful_results,
            "total_fids": total_fids,
            "successful_fids": successful_fids,
            "success_rate": successful_results / total_results if total_results > 0 else 0,
            "total_valid": total_valid,
            "total_invalid": total_invalid,
            "precision": precision
        }
    
    def create_dataframe(self) -> pd.DataFrame:
        """
        Create a DataFrame from dataProvider.validation_batch containing specified columns.
        Only includes results where success is True (Pydantic validation succeeded).
        
        Args:
            dataProvider: DataProvider instance with loaded validation_batch
            
        Returns:
            pd.DataFrame: DataFrame with columns [sid, line_number, feedback, classification]
        """
        rows = []

        # Filter for successful results only
        successful_results = [r for r in self.results if not r.is_failed_sid()]

        for result in successful_results:
            if result.output and result.output.feedback_lines:
                for feedback_line in result.output.feedback_lines:
                    label = None # Default to None if classification is neither valid nor invalid
                    if feedback_line.classification == 'valid':
                        label = 1
                    elif feedback_line.classification == 'invalid':
                        label = 0
                    rows.append({
                        'sid': result.sid,
                        'line_number': feedback_line.line_number,
                        'feedback': feedback_line.feedback,
                        'classification': label
                    })
        
        df = pd.DataFrame(rows)
        df['sid'] = df['sid'].astype(str)
        df['line_number'] = df['line_number'].astype(str)
        df['feedback'] = df['feedback'].astype(str)
        df['classification'] = df['classification'].astype(int)
        print(f"Created DataFrame with {len(df)} rows from {len(successful_results)} successful validation results")

        return df


def validate_validation_output(llm_response: Dict[str, Any]) -> bool:
    """
    Validate that LLM output matches the expected validation format.
    
    Args:
        llm_response: Raw LLM response dictionary
        
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        ValidationOutput(**llm_response)
        return True
    except ValidationError as e:
        print_error(f"Validation output validation failed: {e}")
        return False
