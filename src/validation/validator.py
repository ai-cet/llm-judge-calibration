"""
Refactored validator using Pydantic models and modular design.
"""

import sys
import json
import numpy as np
import pprint
import datetime
from typing import List, Optional
from tqdm import tqdm
import pandas as pd

sys.path.append("..")

from src.config import NUM_SIDS, MODELS_GEN, MODELS_VAL, pathLogs, fpathLLMAsJudge, fpathValidatorSummary, pGa_CONST
from src.validation.data import DataProvider
from src.validation.utils import merge_df_y_yhat, merge_df_merged_yhat, calculate_confusion_matrix_merge, tpr_tnr_list, pretty_print_into_file
from src.validation.ensemble import ensemble_prediction, ensemble_writePg_hat_to_file
from collections import defaultdict


####################
# Config
####################

####################
# Setup
####################

msg = '# AUTO-GENERATED FILE. DO NOT EDIT\n'
with open(fpathLLMAsJudge, 'w') as f:
    f.write(msg)  # Clear the file
with open(fpathValidatorSummary, 'w') as f:
    f.write(msg)  # Clear the file


####################
# LLM-as-a-Judge
####################

def validate_model(models_gen, models_val):
    """
    Validate a specific model and return the results as a batch.
    
    Args:
        modelGen: Model used for generation
        modelVal: Model used for validation
        use_ground_truth: Whether to use ground truth data

    Returns:
        ValidationBatch: The validation results
    """
    error_message_counts = defaultdict(int)
    error_message_counts_validator = defaultdict(lambda: defaultdict(int))
    numFailedFids_max = 0
    label_max = None
    confusion_matrices_gen = []
    confusion_matrices_val = []
    failed_sids = []
    failed_sids_partial = []
    failed_fids = []
    total_fids = []
    merge_failed_fids = []
    GV = []
    dfs = {}
    

    for modelGen in models_gen:
        row_gen = []
        row_gv = []
        failed_sids_row = []
        failed_sids_partial_row = []
        failed_fids_row = []
        total_fids_row = []
        merge_failed_fids_row = []
        df_merged = pd.DataFrame()

        for j, modelVal in enumerate(models_val):
            dataProvider = DataProvider(modelGen, modelVal)

            failedSids = dataProvider.get_failed_sids()
            failedSids_partial = dataProvider.get_failed_sids_partial()
            numFailedFids = dataProvider.get_failed_fids_count()
            numTotalFids = dataProvider.get_total_fids_count()

            if numFailedFids > numFailedFids_max:
                numFailedFids_max = numFailedFids
                label_max = f'{modelGen} -> {modelVal}'

            failed_sids_row += [len(failedSids)]
            failed_sids_partial_row += [len(failedSids_partial)]
            failed_fids_row += [numFailedFids]
            total_fids_row += [numTotalFids]

            # Append error_message_counts
            for key, value in dataProvider.error_message_counts.items():
                error_message_counts[key] += value
                error_message_counts_validator[modelVal][key] += value

            # Create DataFrame with specified columns, filtering for successful validations only
            df_yhat = dataProvider.validation_batch.create_dataframe()
            df_y = dataProvider.generation_batch.create_dataframe()

            df_y.to_csv(f'{pathLogs}/ensemble/df_y.csv', index=False)
            df_yhat.to_csv(f'{pathLogs}/ensemble/df_yhat.csv', index=False)

            # Merge the DataFrames
            if df_merged.empty:
                df_merged = df_y.copy()
            df_merged = merge_df_merged_yhat(df_merged, df_yhat, modelVal)
            numMergeFailedFids = df_merged[df_merged[f'classification_{modelVal}'].isna()].shape[0]
            merge_failed_fids_row.append(numMergeFailedFids)

            # Add confusion matrix calculation
            confusion_matrix = calculate_confusion_matrix_merge(df_y, df_yhat)
            row_gen.append(confusion_matrix)
            if len(confusion_matrices_val) <= j:
                confusion_matrices_val.append([])
            
            prev_row = confusion_matrices_val[j]
            confusion_matrices_val[j] = prev_row + [confusion_matrix]

            tn, fp, fn, tp = confusion_matrix
            percentage_valid = (tp + fp) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0

            print(f"Confusion Matrix for {modelGen} vs {modelVal}: {confusion_matrix}, percentage_valid={percentage_valid:.2f}")
            row_gv.append(percentage_valid)

        df_merged.to_csv(f'{pathLogs}/ensemble/df_{modelGen}.csv', index=False)
        failed_sids.append(failed_sids_row)
        failed_fids.append(failed_fids_row)
        total_fids.append(total_fids_row)
        failed_sids_partial.append(failed_sids_partial_row)
        merge_failed_fids.append(merge_failed_fids_row)

        confusion_matrices_gen.append(row_gen)
        GV.append(row_gv)
        dfs[modelGen] = df_merged

    tprs, tnrs = zip(*(tpr_tnr_list(confusion_matrix) for confusion_matrix in confusion_matrices_val))
    count_total = np.sum(total_fids)
    count_invalids = np.sum(merge_failed_fids)

    return error_message_counts, error_message_counts_validator, failed_sids, failed_sids_partial, failed_fids, total_fids, merge_failed_fids, count_total, count_invalids, confusion_matrices_gen, tprs, tnrs, GV, dfs, label_max


def llm_judge_errors(models_gen, models_val, GV_gen: List[List[float]]):
    max_error_max = 0
    max_error_min = float('inf')
    mean_error_max = 0
    mean_error_min = float('inf')

    for j, modelVal in enumerate(models_val):
        max_error_val = 0
        mean_error_val = 0

        for i, modelGen in enumerate(models_gen):
            valueActual = pGa_CONST[modelGen]
            valuePredicted = GV_gen[i][j]
            error = abs(valueActual - valuePredicted)


            max_error_val = max(max_error_val, error)
            mean_error_val += error

        mean_error_val /= len(models_gen)

        max_error_max = max(max_error_max, max_error_val)
        max_error_min = min(max_error_min, max_error_val)
        mean_error_max = max(mean_error_max, mean_error_val)
        mean_error_min = min(mean_error_min, mean_error_val)

    pretty_print_into_file('llm_judge_max_error_range', (max_error_min, max_error_max), fpathLLMAsJudge, comment='Max error range for LLM-as-a-Judge')
    pretty_print_into_file('llm_judge_mean_error_range', (mean_error_min, mean_error_max), fpathLLMAsJudge, comment='Mean error range for LLM-as-a-Judge')


####################
# Main
####################

if __name__ == "__main__":
    # Route 1
    error_message_counts_gen, error_message_counts_validator_gen, failed_sids_gen, failed_sids_partial_gen, failed_fids_gen, total_fids_gen, merge_failed_fids_gen, count_total_gen, count_invalids_gen, confusion_matrices_gen, tprs_gen, tnrs_gen, GV_gen, dfs_gen, label_max = validate_model(MODELS_GEN, MODELS_VAL)

    # Round 2 for all gens
    error_message_counts_all, error_message_counts_validator_all, failed_sids_all, failed_sids_partial_all, failed_fids_all, total_fids_all, merge_failed_fids_all, count_total_all, count_invalids_all, confusion_matrices_all, tprs_all, tnrs_all, GV_all, dfs_all, label_max = validate_model(MODELS_VAL, MODELS_VAL)

    # Write the stats to files
    pretty_print_into_file('confusion_matrix_validators', confusion_matrices_gen, fpathLLMAsJudge, comment='Confusion Matrix of Annotated Generators by Validators')
    pretty_print_into_file('validator_tpr', tprs_gen, fpathLLMAsJudge, comment='Validator TPRs for Annotated Generators')
    pretty_print_into_file('validator_tnr', tnrs_gen, fpathLLMAsJudge, comment='Validator TNRs for Annotated Generators')
    pretty_print_into_file('GV', GV_all, fpathLLMAsJudge, comment='GV matrix: predicted precision of generators by validators')

    # Calculate the range of errors for LLM-as-a-Judge
    llm_judge_errors(MODELS_GEN, MODELS_VAL, GV_gen)

    # Run the ensemble prediction and write the results
    ensemble_writePg_hat_to_file(dfs_all, MODELS_VAL)
    ensemble_results = ensemble_prediction(dfs_gen, MODELS_GEN, MODELS_VAL)

    # Final summary
    percentage_invalids = round(count_invalids_all / count_total_all * 100, 2)
    pretty_print_into_file('error_message_counts', error_message_counts_all, fpathValidatorSummary, comment='Error message counts during validation')
    pretty_print_into_file('error_message_counts_validator', error_message_counts_validator_all, fpathValidatorSummary, comment='Error message counts per Validator')
    pretty_print_into_file('complete_failed_sids', failed_sids_all, fpathValidatorSummary, comment='SIDs that completely failed validation')
    pretty_print_into_file('partial_failed_sids', failed_sids_partial_all, fpathValidatorSummary, comment='SIDs that partially failed validation')
    pretty_print_into_file('validation_total_fids', total_fids_all, fpathValidatorSummary, comment='Total FIDs after validation')
    pretty_print_into_file('validation_failed_fids', failed_fids_all, fpathValidatorSummary, comment='Failed FIDs after validation')
    pretty_print_into_file('merge_failed_fids', merge_failed_fids_all, fpathValidatorSummary, comment='Failed FIDs after merging DFs')
    pretty_print_into_file('count_invalids_all', count_invalids_all, fpathValidatorSummary, comment='Invalid FIDs by Validators')
    pretty_print_into_file('percentage_invalids', percentage_invalids, fpathValidatorSummary, comment='Percentage of Invalid FIDs by Validators')
    pretty_print_into_file('worst_generator_validator_pair', label_max, fpathValidatorSummary, comment='Worst Generator -> Validator combination')

    # Open and print the contents of the file specified by fpathLLMAsJudge
    print(open(fpathLLMAsJudge, 'r').read())
    print(open(fpathValidatorSummary, 'r').read())
