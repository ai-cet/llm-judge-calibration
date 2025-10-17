import json
import os
import numpy as np
import pandas as pd
import math
import re

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc

# =========================
#   PARSE FEEDBACK
# =========================

def getFeedback(fb):
    if 'feedback' in fb and fb['feedback'] is not None:
        feedback = ' '.join(e for e in fb['feedback'].split())
        feedback = re.sub(r'[^\w\s]', '', feedback)
    else:
        feedback = None

    return feedback

# =========================
#   GOLD LABELLER
# =========================
def labeller_gold(data, model, label):
    # Prepare a list to hold the rows for the DataFrame
    rows = []
    tp, tpe, tpr, fpi, fph, gi = 0, 0, 0, 0, 0, 0

    # Iterate over the data and create rows for each feedback entry
    for entry in data:
        sid = entry['sid']
        for fid, fb in enumerate(entry['feedback'], start=1):
            category = fb['category']
            line_number = str(fb['line_number']) if 'line_number' in fb else str(fb['line_num'])
            feedback = getFeedback(fb)
                
            validity = None

            if category == 'FP-I':
                validity = 0
                fpi += 1
            elif category == 'FP-H':
                validity = 0
                fph += 1
            elif category == 'TP':
                validity = 1
                tp += 1
            elif category == 'FP-E':
                validity = 1
                tpe += 1
            elif category == 'FP-R':
                validity = 1
                tpr += 1
            else:
                continue

            rows.append({
                'generator': model,
                'sid': str(sid),
                'line_number': str(line_number),
                'fid': hash(feedback),
                label: validity
            })

    gi = 100 * (tp + tpe + tpr) / (tp + tpe + tpr + fpi + fph)
    latexStr = f"{model} & {tp} & {tpe} & {tpr} & {fpi} & {fph} & {gi:.1f}\\% \\\\"

    # Create DataFrame
    df = pd.DataFrame(rows)

    return df, latexStr

# =========================
#   OTHER GENERATORS
# =========================

def labeller_gens(data, model):
    # Prepare a list to hold the rows for the DataFrame
    rows = []

    # Iterate over the data and create rows for each feedback entry
    for entry in data:
        sid = entry['sid']
        for fid, fb in enumerate(entry['feedback'], start=1):
            if 'category' in fb:
                category = fb['category']
                if category == 'FN':
                    continue

            line_number = str(fb['line_number']) if 'line_number' in fb else str(fb['line_num'])
            feedback = getFeedback(fb)
            

            rows.append({
                'generator': model,
                'sid': str(sid),
                'line_number': str(line_number),
                'fid': hash(feedback),
            })

    print(f"{model}: {len(rows)} feedbacks")
    # Create DataFrame
    df = pd.DataFrame(rows)

    return df

# =========================
#   GET OTHER GENERATORS
# =========================

def get_gen_data(MODELS, pathGen):
    df_gens = {}

    print('-'*50)
    print(f"Loading generators from {pathGen}")
    print('-'*50)
    for model in MODELS:
        fname = f'{pathGen}/{model}_feedback.json'
        
        data = open(fname).read()
        data = json.loads(data)

        df = labeller_gens(data, model)
        df_gens[model] = df

    return df_gens


# =========================
#   GOLD DATA
# =========================

def get_gold_data(MODELS_GEN, pathGold, fnameGoldTable):
    dfs = {}
    latexStrs = ''

    for model in MODELS_GEN:
        fname = f'{pathGold}/{model}_feedback.json'
    
        data = open(fname).read()
        data = json.loads(data)

        df, latexStr = labeller_gold(data, model, 'y')
        dfs[model] = df
        latexStrs += latexStr + '\n'

    # TODO: Hot fix for gpt-4-turbo
    dfs['gpt-4-turbo'] = dfs['gpt-4-turbo'][dfs['gpt-4-turbo']['sid'] != '338']

    with open(fnameGoldTable, 'w') as f:
        f.write(latexStrs)

    return dfs
    