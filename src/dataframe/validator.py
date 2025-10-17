import json
import os
import numpy as np
import pandas as pd
import math
import re

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc

from dataframe import generator

# =========================
#   VALIDATOR LABELLER
# =========================
def labeller_validator(data, modelG, modelV):
    # Prepare a list to hold the rows for the DataFrame
    rows = []

    # Iterate over the data and create rows for each feedback entry
    for entry in data:
        sid = entry['sid']
        output = entry['output']
        
        if not output or not isinstance(output, dict) or 'feedback_lines' not in output or not output['feedback_lines']:
            continue

        for fid, fb in enumerate(output['feedback_lines'], start=1):
            if 'classification' not in fb or 'feedback' not in fb:
                continue
            if not('line_num' in fb or 'line_number' in fb):
                continue
            
            category = fb['classification']
            line_number = str(fb['line_number']) if 'line_number' in fb else str(fb['line_num'])
            feedback = generator.getFeedback(fb)
                
            validity = None

            if category == 'invalid':
                validity = 0
            elif category == 'valid':
                validity = 1
            else:
                continue

            rows.append({
                'generator': modelG,
                'sid': str(sid),
                'line_number': str(line_number),
                'fid': hash(feedback),
                modelV: validity
            })

    # Create DataFrame
    df = pd.DataFrame(rows)

    return df

# =========================
#   VALIDATE COLUMNS
# =========================
def checkNumCols(MODELS, MODELS_GEN, dfs):
    for gen in MODELS:
        print(f'{gen}: {len(dfs[gen].columns)} columns')

        extraCols = 4
        if gen in MODELS_GEN:
            extraCols = 5
        
        assert len(dfs[gen].columns) == len(MODELS) + extraCols, f"Error: {gen} has {len(dfs[gen].columns)} columns, expected {len(MODELS) + extraCols}"

# =========================
#   MISSING FEEDBACK
# =========================
def checkMissingFeedback(dfs, df_gens):
    print('-'*50)
    print(f"Check for missing validator feedback")
    print('-'*50)

    for gen in dfs.keys():
        countOrig = len(df_gens[gen])
        countValid = len(dfs[gen])

        diff = countOrig - countValid

        print(f"{gen}: {countOrig} - {countValid} = {diff}")


        if diff > 0:
            print(f"\tWarning: {diff} feedback entries are missing for {gen}")

# =========================
#   MISSING LABELS
# =========================
def checkMissingLabels(dfs, df_gens, MODELS):
    print('-'*50)
    print(f"Check for missing validator labels")
    print('-'*50)

    for gen in dfs.keys():
        countOrig = len(df_gens[gen])

        # Count if any column is NaN
        countNA = dfs[gen].isnull().any(axis=1).sum()

        # Count of all cells in the df
        total = len(dfs[gen]) * len(MODELS)
        percentageNA = 100 * countNA / total

        print(f"{gen}: {countNA} / {total} ({percentageNA:.2f}%)")

# =========================
#   GET VALIDATOR
# =========================
def get_validator_data(dfs, df_gens, MODELS, MODELS_GEN, pathValidator):
    print('-'*50)
    print(f"Loading validators from {pathValidator}")
    print('-'*50)

    for i, modelG in enumerate(MODELS):
        for j, modelV in enumerate(MODELS):
            fnames = os.listdir(pathValidator)
            match = f'new_labeller_gen_{modelG}_val_{modelV}_'

            matched_fname = None
            for fname in fnames:
                if match in fname:
                    matched_fname = fname
                    break
            
            if matched_fname:
                data = open(pathValidator + matched_fname).read()
                data = json.loads(data)
                df2 = labeller_validator(data, modelG, modelV)
                
                if modelG in dfs:
                    dfs[modelG] = pd.merge(dfs[modelG], df2, on=['generator', 'sid', 'line_number', 'fid'], how='left')
                else:
                    dfs[modelG] = pd.merge(df_gens[modelG], df2, on=['generator', 'sid', 'line_number', 'fid'], how='left')

                # Drop rows where gold truth doesn't exist
                if modelG in MODELS_GEN:
                    dfs[modelG] = dfs[modelG].dropna(subset=["y"])
            else:
                print(match)

    checkNumCols(MODELS, MODELS_GEN, dfs)
    checkMissingFeedback(dfs, df_gens)
    checkMissingLabels(dfs, df_gens, MODELS)
    return dfs

# =========================
#   GET TPR TNR DF
# =========================

def get_tpr_tnr_df(dfs, MODELS, MODELS_GEN):
    gold_tprs = []
    gold_tnrs = []

    for val in MODELS:
        gold_val_tprs = []
        gold_val_tnrs = []

        for gen in MODELS_GEN:
            df = dfs[gen]
            validCount = len(df[df['y']==1])
            invalidCount = len(df[df['y']==0])

            validCount, invalidCount, validCount / (validCount + invalidCount)

            # Calculate TPR and TNR
            tpr = len(df[(df['y']==1) & (df[val]==1)]) / validCount
            tnr = len(df[(df['y']==0) & (df[val]==0)]) / invalidCount

            gold_val_tprs.append(tpr)
            gold_val_tnrs.append(tnr)

        val_tpr = np.mean(gold_val_tprs)
        val_tnr = np.mean(gold_val_tnrs)
        gold_tprs.append(val_tpr)
        gold_tnrs.append(val_tnr)

    gold_tpr = round(np.mean(gold_tprs),2)
    gold_tnr = round(np.mean(gold_tnrs),2)

    df_val_tpr_tnr = pd.DataFrame({
        'Model': MODELS,
        'TPR': gold_tprs,
        'TNR': gold_tnrs
    })

    print(gold_tpr, gold_tnr)
    return df_val_tpr_tnr

