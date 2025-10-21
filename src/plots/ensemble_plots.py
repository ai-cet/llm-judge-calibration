from src.config import Model, pathImages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from src.config import MODELS_SHORT, MODELS_VAL, VALIDATOR_REPAIR_SUFFIX, ValidatorRepairConfig
from src.plots.validator_plots import get_df_validator_tpr_tnr
from matplotlib.patches import Patch

pathEnsemble = f'{pathImages}/ensemble'

####################
# True Positive Rate vs True Negative Rate Plot
####################
def get_df_ensemble_tpr_tnr(ensemble_tpr, ensemble_tnr, match):
    rows = []

    for key in ensemble_tpr.keys():
        if key.startswith(match):
            tpr = ensemble_tpr[key] * 100  # Convert to percentage
            tnr = ensemble_tnr[key] * 100  # Convert to percentage
            rows.append({'ensemble': int(key[1:]), 'tpr': tpr, 'tnr': tnr})

    rows.sort(key=lambda x: x['ensemble'])  # Sort by ensemble index

    df = pd.DataFrame(rows)
    return df

def tpr_tnr_ensemble(ensemble_tpr, ensemble_tnr, validator_tpr, validator_tnr):
    '''Plot the TPR and TNR for the ensemble models.'''
    # Parse the data
    df_v = get_df_ensemble_tpr_tnr(ensemble_tpr, ensemble_tnr, 'v')
    df_i = get_df_ensemble_tpr_tnr(ensemble_tpr, ensemble_tnr, 'i')
    df_validator = get_df_validator_tpr_tnr(validator_tpr, validator_tnr)

    # Create the plot
    plt.figure(figsize=(8, 4))
    plt.plot(df_v['tpr'], df_v['tnr'], marker='^', linewidth=2, label='valid voting', color='blue')
    plt.plot(df_i['tpr'], df_i['tnr'], marker='v', linewidth=2, label='invalid voting', color='red')
    plt.scatter(df_validator['tpr'], df_validator['tnr'], marker='o', color='green', s=20, label='validator models')

    plt.legend(['Valid Voting', 'Invalid Voting', 'LLM as Judge'], fontsize=12)

    # Add small labels for valid and invalid voting next to markers.
    for i, row in df_v.iterrows():
        plt.text(row['tpr'] - 1, row['tnr'] - 8, f'v{int(row["ensemble"])}', fontsize=10, color='black', ha='left')
    for i, row in df_i.iterrows():
        plt.text(row['tpr'] + 1, row['tnr'] + 5, f'i{int(row["ensemble"])}', fontsize=10, color='black', ha='right')

    plt.ylim(0,100)
    plt.xlim(40,100)

    # Labels and fontsize
    plt.xlabel('True Positive Rate %', fontsize=14)
    plt.ylabel('True Negative Rate %', fontsize=14)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.tight_layout()
    plt.savefig(f'{pathEnsemble}/tpr_tnr_ensemble{VALIDATOR_REPAIR_SUFFIX}.pdf', bbox_inches='tight')


####################
# Valid and Invalid Voting Errors
####################
def valid_invalid_error(ensemble_max_errors):
    '''Plot the valid and invalid voting errors from the ensemble summary data.'''

    # Parse the data
    v_data = []
    i_data = []

    for key, value in ensemble_max_errors.items():
        value = value * 100  # Convert to percentage
        if key.startswith('v'):
            v_data.append({'index': int(key[1:]), 'value': value})
        elif key.startswith('i'):
            i_data.append({'index': int(key[1:]), 'value': value})

    # Sort the data by index
    v_data.sort(key=lambda x: x['index'])
    i_data.sort(key=lambda x: x['index'])

    # Create dataframes
    df_v = pd.DataFrame(v_data)
    df_i = pd.DataFrame(i_data)

    # Create the plot
    plt.figure(figsize=(8, 4))
    plt.plot(df_i['index'], df_i['value'], marker='s', linewidth=2, label='Invalid Voting', color='red')
    plt.plot(df_v['index'], df_v['value'], marker='o', linewidth=2, label='Valid Voting', color='blue')

    plt.legend(['Invalid Voting', 'Valid Voting'], fontsize=12)

    # Adjust the labels and sizing
    plt.xlabel('Number of Votes Required to Cross Threshold', fontsize=16)
    plt.ylabel('Maximum Absolute Error (%)', fontsize=16)
    plt.legend(fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.xticks(range(1, 15), fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylim(0, 50)  # Set y-axis limit to 0-50% for better visibility
    plt.tight_layout()
    plt.savefig(f'{pathEnsemble}/valid_invalid_error{VALIDATOR_REPAIR_SUFFIX}.pdf')
