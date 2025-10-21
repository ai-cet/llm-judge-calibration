from statistics import mean

import pandas as pd
from src.config import NUM_SIDS, MODELS_VAL, pGa_CONST, pathOutput, fpathLLMAsJudge, fpathValidatorSummary, fpathEnsembleResults
from src.validation.utils import ppv_npv, pretty_print_into_file, calculate_confusion_matrix, tpr_tnr

####################
# Voting ensemble
####################
def invalid_voting(df, validation_columns, invalid_count):
    """
    Perform invalid voting based on the validation columns.
    
    Args:
        df: DataFrame containing validation results
        validation_columns: List of columns to consider for voting
    
    Returns:
        Series: Invalid votes count for each row
    """
    # Count how many models predict invalid (0) for each row
    invalid_votes = (df[validation_columns] == 0).sum(axis=1)
    
    # Ensemble rule: predict invalid (0) if at least 4 models predict invalid (0)
    df[f'ensemble_i{invalid_count}'] = (invalid_votes >= invalid_count).astype(int)
    
    # Flip the logic: when enough models say invalid (0), we predict invalid (0)
    df[f'ensemble_i{invalid_count}'] = 1 - df[f'ensemble_i{invalid_count}']

    return df

def valid_voting(df, validation_columns, valid_count):
    """
    Perform valid voting based on the validation columns.
    
    Args:
        df: DataFrame containing validation results
        validation_columns: List of columns to consider for voting
    
    Returns:
        Series: Valid votes count for each row
    """
    # Count how many models predict valid (1) for each row
    valid_votes = (df[validation_columns] == 1).sum(axis=1)
    
    # Ensemble rule: predict valid (1) if at least 4 models predict valid (1)
    df[f'ensemble_v{valid_count}'] = (valid_votes >= valid_count).astype(int)

    return df

def calculate_ensemble_prediction(df, ensemble_label):
    """
    Calculate ensemble accuracy for a specific model generation.

    Args:
        df: DataFrame containing validation results
        ensemble_label: Ensemble label for the model generation

    Returns:
        float: Ensemble accuracy for the model generation
    """
    # Calculate confusion matrix for ensemble vs ground truth
    y_true = df['classification']
    y_pred = df[ensemble_label]
    
    # Remove NaN values
    # mask = ~(y_true.isna() | y_pred.isna())
    # y_true = y_true[mask]
    # y_pred = y_pred[mask]
    
    # Calculate confusion matrix components
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    tp = ((y_true == 1) & (y_pred == 1)).sum()

    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Predicted error
    predScore = (tp + fp) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    return predScore
    
####################
# Ensemble TPR and TNR
####################
def init_ensemble_dictionary(length=len(MODELS_VAL) + 1, defaultValue=0):
    hash = {f'v{i}': defaultValue for i in range(1, length)}
    hash.update({f'i{i}': defaultValue for i in range(1, length)})

    return hash

def write_tpr_tnr_to_file(dfs_ensemble):
    ensemble_tpr = init_ensemble_dictionary()
    ensemble_tnr = init_ensemble_dictionary()
    ensemble_ppv = init_ensemble_dictionary()
    ensemble_npv = init_ensemble_dictionary()

    # Unify the df
    df = pd.concat(dfs_ensemble.values(), ignore_index=True)
    print(df)

    for ensembleModel in ensemble_tpr.keys():
        # Calculate confusion matrix for the ensemble model
        y_true = df['y']
        y_pred = df[f'ensemble_{ensembleModel}']
        confusion_matrix = calculate_confusion_matrix(y_true, y_pred)

        # Calculate TPR and TNR
        tn, fp, fn, tp = confusion_matrix
        tpr, tnr = tpr_tnr(tn, fp, fn, tp)
        ppv, npv = ppv_npv(tn, fp, fn, tp)

        # Store TPR and TNR in the dictionary
        ensemble_tpr[ensembleModel] = tpr
        ensemble_tnr[ensembleModel] = tnr
        ensemble_ppv[ensembleModel] = ppv
        ensemble_npv[ensembleModel] = npv

    # Write to summary file
    pretty_print_into_file('ensemble_tpr', ensemble_tpr, fpathValidatorSummary, comment='Ensemble TPR')
    pretty_print_into_file('ensemble_tnr', ensemble_tnr, fpathValidatorSummary, comment='Ensemble TNR')
    pretty_print_into_file('ensemble_ppv', ensemble_ppv, fpathValidatorSummary, comment='Ensemble PPV')
    pretty_print_into_file('ensemble_npv', ensemble_npv, fpathValidatorSummary, comment='Ensemble NPV')

####################
# Ensemble Prediction
####################

def ensemble_writePg_hat_to_file(dfs_all, MODEL_VALS):
    """
    Write the predicted precisions to a file.
    
    Args:
        pG_hat: Dictionary containing predicted precisions
        fpath: File path to write the results
    """
    # Get model validation columns (exclude 'classification' which is ground truth)
    validation_columns = [f'classification_{model}' for model in MODEL_VALS]

    # Compute predicted precisions for all models
    pG_hat = []
    for model in MODEL_VALS:
        df = dfs_all[model].copy()
        i = 4
        df = invalid_voting(df, validation_columns, i)  # Perform invalid voting
        predScore_i = calculate_ensemble_prediction(df, f'ensemble_i{i}')
        pG_hat.append(predScore_i)

    # Print the predicted precisions
    pretty_print_into_file('ensemble_predicted_precisions', pG_hat, fpathLLMAsJudge, comment='Ensemble Predicted Precisions')

def ensemble_prediction(dfs_gen, MODEL_GENS, MODEL_VALS, valid_count=None, invalid_count=None):
    """
    Create ensemble predictions and calculate errors for each dataset.
    
    Args:
        dfs_gen: Dictionary of dataframes for different generator models
    
    Returns:
        Dictionary containing ensemble results and errors for each model
    """
    # Get model validation columns (exclude 'classification' which is ground truth)
    validation_columns = [f'classification_{model}' for model in MODEL_VALS]
    num_validators = len(validation_columns)
    dfs_ensemble = {}

    # Initialize vars to store max and mean errors for the majority and best models
    majority_max_error = None
    majority_mean_error = None
    best_max_error = None
    best_mean_error = None
    
    # Initialize max and mean errors for each ensemble
    ensemble_max_errors = init_ensemble_dictionary(length=num_validators + 1)
    ensemble_mean_errors = init_ensemble_dictionary(length=num_validators + 1)
    ensemble_max_errors_modelGen = init_ensemble_dictionary(length=num_validators + 1, defaultValue=None)
    
    # Compute max and mean errors for the majority and best models
    for model_gen in MODEL_GENS:
        df = dfs_gen[model_gen].copy()

        # Perform valid and invalid voting for each validator
        for i in range(1, num_validators + 1):
            df = valid_voting(df, validation_columns, i)
            predScore_v = calculate_ensemble_prediction(df, f'ensemble_v{i}')
            error_v = abs(predScore_v - pGa_CONST[model_gen])
            max_error_v = max(ensemble_max_errors[f'v{i}'], error_v)
            ensemble_max_errors[f'v{i}'] = max_error_v
            ensemble_max_errors_modelGen[f'v{i}'] = model_gen if max_error_v == error_v else ensemble_max_errors_modelGen[f'v{i}']
            ensemble_mean_errors[f'v{i}'] += error_v

            df = invalid_voting(df, validation_columns, i)
            predScore_i = calculate_ensemble_prediction(df, f'ensemble_i{i}')
            error_i = abs(predScore_i - pGa_CONST[model_gen])
            max_error_i = max(ensemble_max_errors[f'i{i}'], error_i)
            ensemble_max_errors[f'i{i}'] = max_error_i
            ensemble_max_errors_modelGen[f'i{i}'] = model_gen if max_error_i == error_i else ensemble_max_errors_modelGen[f'i{i}']
            ensemble_mean_errors[f'i{i}'] += error_i

        # Replace column names that start with 'classification' to start with 'y'
        df.columns = [col.replace('classification', 'y') if col.startswith('classification') else col for col in df.columns]

        # Store the ensemble results for this model generation
        dfs_ensemble[model_gen] = df
        
    # Save ensemble results to Excel
    with pd.ExcelWriter(fpathEnsembleResults) as writer:
        for model_gen in MODEL_GENS:
            df = dfs_ensemble[model_gen]
            df.to_excel(writer, sheet_name=model_gen, index=False)

    # Calculate mean errors
    for key in ensemble_mean_errors:
        ensemble_mean_errors[key] /= len(MODEL_GENS)

    # Calculate the majority 
    majority_i = num_validators // 2 + 1 # if num_validators % 2 == 1 else num_validators // 2
    majority_max_error = ensemble_max_errors[f'v{majority_i}']
    majority_mean_error = ensemble_mean_errors[f'v{majority_i}']

    # Calculate the best (least) max and mean error among all models
    best_i = min(ensemble_max_errors, key=ensemble_max_errors.get)
    best_max_error = ensemble_max_errors[best_i]
    best_mean_error = ensemble_mean_errors[best_i]

    # Write TPR and TNR to file
    write_tpr_tnr_to_file(dfs_ensemble)
    
    # Print results to file
    pretty_print_into_file('ensemble_majority_max_error', majority_max_error, fpathLLMAsJudge, comment='Max error range for the majority ensemble')
    pretty_print_into_file('ensemble_majority_mean_error', majority_mean_error, fpathLLMAsJudge, comment='Mean error range for the majority ensemble')
    pretty_print_into_file('ensemble_best_max_error', best_max_error, fpathLLMAsJudge, comment='Max error range for the best model')
    pretty_print_into_file('ensemble_best_mean_error', best_mean_error, fpathLLMAsJudge, comment='Mean error range for the best model')

    # Print the summary for checks
    pretty_print_into_file('ensemble_max_errors', ensemble_max_errors, fpathValidatorSummary, comment='Ensemble Max Errors')
    pretty_print_into_file('ensemble_mean_errors', ensemble_mean_errors, fpathValidatorSummary, comment='Ensemble Mean Errors')
