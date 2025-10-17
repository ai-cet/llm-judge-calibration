from src.config import Model, pathImages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from src.config import MODELS_SHORT, MODELS_VAL, VALIDATOR_REPAIR_SUFFIX, ValidatorRepairConfig
from src.plots.validator_plots import get_df_validator_tpr_tnr
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

pathEnsemble = f'{pathImages}/ensemble'

####################
# Data
####################
def get_repair_suffixes():
    # List of flags in ValidatorRepairConfig
    VALIDATOR_REPAIR_FLAGS = ValidatorRepairConfig.flags()
    validator_repair = ValidatorRepairConfig()
    
    # Set them false initially and append to suffixes
    for flag in VALIDATOR_REPAIR_FLAGS:
        setattr(validator_repair, flag, False)
    suffixNoRepair = validator_repair.getSuffix()

    # Set them all to true initially and append to suffixes
    for flag in VALIDATOR_REPAIR_FLAGS:
        setattr(validator_repair, flag, True)
    suffixAllTrue = validator_repair.getSuffix()

    # Cumulatively, grow the suffixes with each flag set to True
    suffixes = [suffixNoRepair]
    suffixCummulative = suffixAllTrue[0]
    for key in suffixAllTrue[1:]:
        suffixCummulative += key
        suffixes.append(suffixCummulative)

    return suffixes

def get_repair_ensemble_data(suffix):
    try:
        # Load the module dynamically based on the suffix
        module = __import__(f'src.validation.generated_scripts.summary{suffix}', fromlist=['ensemble_max_errors', 'count_invalids_all', 'percentage_invalids'])
        ensemble_max_errors = module.ensemble_max_errors
        count_invalids_all = module.count_invalids_all
        percentage_invalids = module.percentage_invalids
        # If success
        if ensemble_max_errors:
            repair = ValidatorRepairConfig.getName(suffix)
            ensemble_max_error_majority = ensemble_max_errors['v8'] * 100  # Majority voting error
            ensemble_max_error_best = min(ensemble_max_errors.values()) * 100  # Best voting error
            # If the best error is not available, set it to 0            

            # Attach to DF rows
            row = [f'{repair}', count_invalids_all, percentage_invalids, ensemble_max_error_majority, ensemble_max_error_best]
            return row
    except ImportError as e:
        pass
        # print(f"Error importing module for suffix {suffix}: {e}")

def get_repair_regression_data(suffix):
    '''Load the regression data for a given repair suffix.'''
    try:
        # Load the module dynamically based on the suffix
        module = __import__(f'src.regression.generated_scripts.summary{suffix}', fromlist=['regression_max_errors'])
        regression_max_errors = module.regression_max_errors
        if regression_max_errors:
            row = []

            # Get the max error for the regression @ k=1, 3 and 5 
            for k, v in enumerate(regression_max_errors):
                if k in [0, 1, 3, 5]:
                    meanMaxError = np.mean(v)
                    row.append(meanMaxError)

            return row
    except ImportError as e:
        pass
        # print(f"Error importing module for suffix {suffix}: {e}")

def get_repair_df(suffixes):
    '''Load the ensemble max errors for a given repair suffix.'''
    rows = []

    for suffix in suffixes:
        rowEnsemble = get_repair_ensemble_data(suffix)
        rowRegression = get_repair_regression_data(suffix)
        if rowEnsemble and rowRegression:
            rows.append(rowEnsemble + rowRegression)

    df = pd.DataFrame(rows, columns=['repair', 'count_invalids_all', 'percentage_invalids', 'error_majority', 'error_best', 
                                     'regression_error_0', 'regression_error_1', 'regression_error_3', 'regression_error_5'])
    return df

####################
# Plotting
####################
def plot_repair_vs_ensemble(df):
    '''Plot the repair vs ensemble errors. On x-axis, we have the repair configurations, and on y1-axis, we have the count of invalids, and on y2-axis, we have the error percentages.'''
    # Set the figure size
    plt.figure(figsize=(8, 6.25))

    # Create a twin axis for the count of invalids
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    # Plot the count of invalids on the first y-axis
    sns.barplot(x='repair', y='percentage_invalids', data=df, ax=ax1, color='lightgray', alpha=0.7)
    ax1.set_ylabel('Missing Values (%)', fontsize=18)
    ax1.set_xlabel('Repairs Undertaken', fontsize=18)
    ax1.tick_params(axis='y', labelsize=14)

    # Plot the error percentages on the second y-axis
    sns.lineplot(x='repair', y='error_majority', data=df, ax=ax2, marker='^', color='brown')
    sns.lineplot(x='repair', y='error_best', data=df, ax=ax2, marker='o', color='blue', dashes=(4, 4), linestyle=':')

    # Plot the regression errors
    sns.lineplot(x='repair', y='regression_error_0', data=df, ax=ax2, marker='x', color='black', linestyle='-', markeredgecolor='black')
    sns.lineplot(x='repair', y='regression_error_1', data=df, ax=ax2, marker='s', color='green', dashes=(3, 1), linestyle='--')
    sns.lineplot(x='repair', y='regression_error_3', data=df, ax=ax2, marker='d', color='red', dashes=(5, 2), linestyle='-.')
    sns.lineplot(x='repair', y='regression_error_5', data=df, ax=ax2, marker='*', color='purple', dashes=(2, 2), linestyle='--', markersize=10)

    ax2.set_ylabel('Maximum Absolute Error (%)', fontsize=18)
    ax2.tick_params(axis='y', labelsize=12)

    # Add a legend with a grey box for the barplot
    legend_elements = [
        plt.Line2D([0], [0], color='black', marker='x', label='Regression @ s=0', linestyle='-', markersize=10, ),

        plt.Line2D([0], [0], color='brown', marker='^', label='Majority Consensus'),
        plt.Line2D([0], [0], color='blue', marker='o', label='Minority Veto', linestyle=':'),

        plt.Line2D([0], [0], color='green', marker='s', label='Regression @ s=1', linestyle='--'),
        plt.Line2D([0], [0], color='red', marker='d', label='Regression @ s=3', linestyle='-.'),
        plt.Line2D([0], [0], color='purple', marker='*', label='Regression @ s=5', dashes=(2, 2), linestyle='--', markersize=10),
        Patch(facecolor='lightgray', label='Missing Values')
    ]
    ax2.legend(handles=legend_elements, fontsize=16, loc='upper center', bbox_to_anchor=(0.5, 1.45), ncol=2)

    # Set the x-ticks to be the repair configurations
    xticks = ['+ ' + i if i != 'Original' else i for i in df['repair'].tolist()]
    ax1.set_xticks(range(len(xticks)))
    ax1.set_xticklabels(xticks, fontsize=14) #  rotation=45, ha='right',
    ax1.set_ylim(0, 18)
    ax2.set_ylim(0, 18)  # Set y-axis limit to 0-18 for better visibility

    # Convert to ints
    ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True))

    ticks = ax1.get_yticks()
    ax1.set_yticks(ticks)
    ax1.set_yticklabels([f'{int(y)}' for y in ticks], fontsize=14)
    ax2.set_yticks(ticks)
    ax2.set_yticklabels([f'{int(y)}' for y in ticks], fontsize=14)

    # Ensure both axes have the same y-axis ticks
    ax2.set_yticklabels(ax1.get_yticks(), fontsize=14)
    ax1.set_yticks(ax2.get_yticks())
    ax1.set_yticklabels(ax2.get_yticks(), fontsize=14)

    # Add a legend
    # ax2.legend(['Majority Error', 'Best Error'], fontsize=12, loc='upper left')

    plt.tight_layout()
    plt.savefig(f'{pathEnsemble}/repair_vs_ensemble_error.pdf', bbox_inches='tight')

####################
# Main
####################

def repair_vs_error():
    '''Plot the repair vs ensemble errors.'''

    # Get the suffixes for the repair configurations
    suffixes = get_repair_suffixes()

    # Load the DF for each suffix
    df = get_repair_df(suffixes)

    # Plot the data
    plot_repair_vs_ensemble(df)