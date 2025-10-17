import pprint
from typing import List
import pandas as pd
import json
import numpy as np

####################
# DF mergers
####################
def merge_df_y_yhat(df_y: pd.DataFrame, df_yhat: pd.DataFrame) -> pd.DataFrame:
    # Convert key columns to string type for consistent merging
    df_yhat = df_yhat.copy()
    df_yhat['sid'] = df_yhat['sid'].astype(str)
    df_yhat['line_number'] = df_yhat['line_number'].astype(str) 
    df_yhat['feedback'] = df_yhat['feedback'].astype(str)

    df_y['sid'] = df_y['sid'].astype(str)
    df_y['line_number'] = df_y['line_number'].astype(str)
    df_y['feedback'] = df_y['feedback'].astype(str)

    merged_df = df_y.merge(
        df_yhat, 
        on=['sid', 'line_number', 'feedback'], 
        how='left', 
        suffixes=('_true', '_pred')
    )
    return merged_df

def merge_df_merged_yhat(df_merged: pd.DataFrame, df_yhat: pd.DataFrame, model) -> pd.DataFrame:
    # Convert key columns to string type for consistent merging
    df_merged['sid'] = df_merged['sid'].astype(str)
    df_merged['line_number'] = df_merged['line_number'].astype(str)
    df_merged['feedback'] = df_merged['feedback'].astype(str)

    df_yhat = df_yhat.copy()
    df_yhat['sid'] = df_yhat['sid'].astype(str)
    df_yhat['line_number'] = df_yhat['line_number'].astype(str) 
    df_yhat['feedback'] = df_yhat['feedback'].astype(str)

    
    merged_df = pd.merge(
        df_merged, 
        df_yhat, 
        on=['sid', 'line_number', 'feedback'], 
        how='left', 
        suffixes=('', f'_{model}')
    )
    return merged_df


####################
# TPR and TNR calculations
####################
def calculate_confusion_matrix(y_true: pd.Series, y_pred: pd.Series) -> List[int]:
    """
    Calculate confusion matrix [TN, FN, FP, TP] from ground truth and predictions.
    
    Args:
        y_true: Ground truth labels as a pandas Series
        y_pred: Predicted labels as a pandas Series
    
    Returns:
        List[int]: [TN, FN, FP, TP] counts
    """
    # Calculate confusion matrix components
    tn = ((y_true == 0) & (y_pred == 0)).sum()  # True Negative
    fp = ((y_true == 0) & (y_pred == 1)).sum()  # False Positive
    fn = ((y_true == 1) & (y_pred == 0)).sum()  # False Negative
    tp = ((y_true == 1) & (y_pred == 1)).sum()  # True Positive
    
    return [tn, fp, fn, tp]


def calculate_confusion_matrix_merge(df_y: pd.DataFrame, df_yhat: pd.DataFrame) -> List[int]:
    """
    Calculate confusion matrix [TN, FN, FP, TP] from ground truth and predictions.
    
    Args:
        df_y: Ground truth DataFrame with columns [sid, line_number, feedback, classification]
        df_yhat: Predictions DataFrame with columns [sid, line_number, feedback, classification]
    
    Returns:
        List[int]: [TN, FN, FP, TP] counts
    """
    # Perform left outer join on sid and line_number
    merged_df = merge_df_y_yhat(df_y, df_yhat)

    y_true = merged_df['classification_true']
    y_pred = merged_df['classification_pred']

    return calculate_confusion_matrix(y_true, y_pred)

def tpr_tnr(tn, fp, fn, tp):
    """
    Calculate True Positive Rate (TPR) and True Negative Rate (TNR).
    
    Args:
        tn: True Negatives
        fp: False Positives
        fn: False Negatives
        tp: True Positives

    Returns:
        Tuple[float, float]: (TPR, TNR)
    """
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
    return tpr, tnr

def ppv_npv(tn, fp, fn, tp):
    """
    Calculate Positive Predictive Value (PPV) and Negative Predictive Value (NPV).
    
    Args:
        tn: True Negatives
        fp: False Positives
        fn: False Negatives
        tp: True Positives

    Returns:
        Tuple[float, float]: (PPV, NPV)
    """
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    return ppv, npv

def tpr_tnr_list(confusion_matrices):

    """
    Calculate TPR and TNR for a list of confusion matrices.
    
    Args:
        confusion_matrices: List of confusion matrices [TN, FP, FN, TP]
    Returns:

        List[Tuple[float, float]]: List of (TPR, TNR) tuples
    """
    tn_cumulative = 0
    fp_cumulative = 0
    fn_cumulative = 0
    tp_cumulative = 0
    

    for cm in confusion_matrices:
        tn, fp, fn, tp = cm
        tn_cumulative += tn
        fp_cumulative += fp
        fn_cumulative += fn
        tp_cumulative += tp

    tpr, tnr = tpr_tnr(tn_cumulative, fp_cumulative, fn_cumulative, tp_cumulative)
    return (tpr, tnr)


####################
# Printing
####################

def default_print(x):
    '''Convert numpy integers to Python int'''
    return int(x) if isinstance(x, (np.integer, np.int64)) else str(x)

def pretty_print_into_file(varName, var, fpath, comment = ''):
    """
    Pretty print the contents of a variable into a specified path.

    Args:
        var: The variable to be pretty printed.
    """
    with open(fpath, 'a') as f:
        # print(var)
        value = json.dumps(var, indent=2, # Use JSON for serialization
                    default=default_print) # Convert numpy integers to Python int
        value = value.replace('NaN', 'None')  # Replace NaN with None for JSON compatibility
        # print(value)
        f.write(f"\n# {comment}\n{varName} = {value}\n")
        