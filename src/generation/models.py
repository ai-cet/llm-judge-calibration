"""
Pydantic models for Generator data validation.
"""

from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, ValidationError, validator, Field, model_validator
from src.utils import print_warning, print_error
import pandas as pd
import os


class LLMFeedback(BaseModel):
    """Model for individual feedback from LLM output."""
    line_number: Union[str, int] = Field(..., description="Line number where the mistake occurs")
    feedback: str = Field(..., description="Feedback text for the student")
    
    @validator('line_number', pre=True)
    def validate_line_number(cls, v):
        """Convert line_number to string for consistency."""
        return str(v)


class LLMOutput(BaseModel):
    """Model for raw LLM output format."""
    correct_code: str = Field(..., description="The student code with minimal changes to pass all test cases")
    feedbacks: List[LLMFeedback] = Field(..., description="List of feedback items")


class ProcessedFeedback(BaseModel):
    """Model for processed feedback with category."""
    line_number: Union[str, int] = Field(..., description="Line number where the mistake occurs")
    feedback: str = Field(..., description="Feedback text for the student")
    category: Optional[str] = Field(None, description="Feedback category (TP, FP-H, FP-I, FP-E, FN)")
    classification: Optional[str] = Field(None, description="Feedback validity classification (valid/invalid)")
    
    @validator('category')
    def validate_category(cls, v):
        """Ignore feedback items with category 'FN'."""
        if v == 'FN':
            raise ValueError("Feedback items with category 'FN' should be ignored")
        return v
    
    @validator('classification', pre=True, always=True)
    def set_classification_from_category(cls, v, values):
        """Set classification based on category."""
        category = values.get('category')
        if category in ['TP', 'TP-R', 'FP-R', 'TP-E', 'FP-E']:
            return 'valid'
        elif category in ['FP-H', 'FP-I']:
            return 'invalid'
        else:
            # If category is not in either list, keep original value or default
            return v if v is not None else 'valid'


class RepairSuccess(BaseModel):
    """Model for repair success status."""
    success: bool = Field(..., description="Whether the repair was successful")
    status_code: int = Field(..., description="Status code of the repair operation")

class GeneratorData(BaseModel):
    """Model for complete Generator data structure."""
    sid: int = Field(..., description="Student ID")
    repaired_code: str = Field(..., description="Code after repair (equal to correct_code from LLM)")
    feedback: List[ProcessedFeedback] = Field(..., description="List of processed feedback with categories")
    student_code: str = Field(..., description="Original student code from dataset")
    pid: int = Field(..., description="Problem ID")
    category_required: bool = Field(default=True, description="Whether category field is required in feedback")

    def __init__(self, **data):
        """Generate feedback list without 'FN' category."""
        if 'feedback' in data:
            feedback_list = data['feedback']
            if isinstance(feedback_list, list):
                # Remove feedback items with category 'FN' before validation
                filtered_feedback = []
                for item in feedback_list:
                    if isinstance(item, dict) and item.get('category') != 'FN':
                        filtered_feedback.append(item)
                data['feedback'] = filtered_feedback
        super().__init__(**data)
    
    @model_validator(mode='before')
    def validate_feedback_with_category_required(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate feedback list based on category_required setting."""
        feedback_list = values.get('feedback', [])
        category_required = values.get('category_required', True)
        
        if not isinstance(feedback_list, list):
            raise ValueError("feedback must be a list")
        
        for item in feedback_list:
            if not isinstance(item, dict):
                raise ValueError("Each feedback item must be a dictionary")
            
            required_fields = ['line_number', 'feedback']
            if category_required:
                required_fields.append('category')
            
            for field in required_fields:
                if field not in item:
                    raise ValueError(f"Missing required field '{field}' in feedback item")
            
            # Additional check for empty category when required
            if category_required and (not item.get('category') or item.get('category') == ''):
                raise ValueError("Category cannot be empty when category_required is True")
        
        return values
    
class GenerationBatch(BaseModel):
    """Model for a batch of Generator data."""
    generator_model: str = Field(..., description="Model used for generation")
    results: List[GeneratorData] = Field(..., description="List of Generator data entries")
    use_ground_truth: bool = Field(default=False, description="Whether to use ground truth data for validation")
    
    @model_validator(mode='before')
    def validate_results(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure results is a list of GeneratorData."""
        results = values.get('results', [])
        if not isinstance(results, list):
            raise ValueError("results must be a list")
        
        # Validate each result
        for item in results:
            if not isinstance(item, dict):
                raise ValueError("Each result must be a dictionary")
            try:
                GeneratorData(**item)
            except ValidationError as e:
                raise ValueError(f"Invalid GeneratorData entry")
        
        return values
    
    def getById(self, sid: int) -> Optional['GeneratorData']:
        """
        Retrieve a GeneratorData by student ID.

        Args:
            sid: Student ID as integer

        Returns:
            GenerationBatch or None: Batch if found, None otherwise
        """
        for item in self.results:
            if item.sid == sid:
                return item
        return None
    
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

        for result in self.results:
            if result.feedback:
                for feedback_line in result.feedback:
                    rows.append({
                        'sid': result.sid,
                        'line_number': feedback_line.line_number,
                        'feedback': feedback_line.feedback,
                        'classification': 1 if feedback_line.classification == 'valid' else 0
                    })
        
        df = pd.DataFrame(rows)
        df['sid'] = df['sid'].astype(str)
        df['line_number'] = df['line_number'].astype(str)
        df['feedback'] = df['feedback'].astype(str)
        df['classification'] = df['classification'].astype(int)
        return df

def validate_llm_output(llm_response: Dict[str, Any]) -> bool:
    """
    Validate that LLM output matches the expected format.
    
    Args:
        llm_response: Raw LLM response dictionary
        
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        LLMOutput(**llm_response)
        return True
    except ValidationError as e:
        print_error(f"LLM output validation failed: {e}")
        return False


def convert_llm_to_generator_data(llm_output: Dict[str, Any], sid: int, pid: int, 
                                student_code_mapping: Dict[int, str], category_required: bool = False) -> Optional[Dict[str, Any]]:
    """
    Convert LLM output to Generator data format with student code from dataset.
    
    Args:
        llm_output: Raw LLM output
        sid: Student ID
        pid: Problem ID
        student_code_mapping: Mapping of sid to student code
        category_required: Whether category field is required in feedback
        
    Returns:
        Dict or None: Converted data if successful, None otherwise
    """
    try:
        # Validate LLM output first
        if not validate_llm_output(llm_output):
            return None
        
        # Get student code from mapping
        student_code = student_code_mapping.get(sid)
        if student_code is None:
            print_warning(f"No student code found for sid {sid}")
            return None
        
        # Convert feedbacks to processed format (without categories initially)
        processed_feedback = []
        for fb in llm_output['feedbacks']:
            feedback_data = {
                'line_number': str(fb['line_number']),
                'feedback': fb['feedback']
            }
            
            # Only add category if it's required
            if category_required:
                feedback_data['category'] = ''  # Default category, should be updated by labeling process
            
            processed_feedback.append(feedback_data)
        
        # Create generator data structure
        generator_data = {
            'sid': sid,
            'repaired_code': llm_output['correct_code'],
            'feedback': processed_feedback,
            'student_code': student_code,
            'pid': pid,
            'category_required': category_required
        }
        
        # Validate the complete structure
        if validate_generator_data(generator_data, category_required):
            return generator_data
        else:
            return None
            
    except Exception as e:
        print_error(f"Error converting LLM output for sid {sid}: {e}")
        return None


