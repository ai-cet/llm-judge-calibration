"""
Validation module for LLM feedback assessment.

This module provides a modular, Pydantic-based approach to validating
LLM-generated feedback using various validation models.
"""

from .models import (
    ValidationOutput, 
    ValidationResult,
    ValidationBatch,
    ValidatedFeedbackLine,
    GroundTruthFeedback
)

from .service import (
    ValidationService,
    ValidationRunner,
    create_attempt_name,
    calculate_precision_stats,
    print_validation_stats
)

from .data import (
    DataProvider
)

from .prompt import (
    get_validation_prompt,
    get_validation_prompt_with_ground_truth
)

__all__ = [
    # Models
    'ValidationOutput',
    'ValidationResult', 
    'ValidationBatch',
    'ValidatedFeedbackLine',
    'GroundTruthFeedback',

    # Services
    'ValidationService',
    'ValidationRunner',
    'create_attempt_name',
    'calculate_precision_stats',
    'print_validation_stats',
    
    # Data access
    'DataProvider',
   
    # Utilities
    
    # Prompts
    'get_validation_prompt',
    'get_validation_prompt_with_ground_truth'
]