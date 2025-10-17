from src.config import Model, pathImages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from src.config import MODELS_SHORT, MODELS_VAL, VALIDATOR_REPAIR_SUFFIX, ValidatorRepairConfig
from src.plots.validator_plots import get_df_validator_tpr_tnr
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, FormatStrFormatter


pathRegression = f'{pathImages}/regression'

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
        print(f"Error importing module for suffix {suffix}: {e}")

def get_repair_regression_data(suffix):
    '''Load the regression data for a given repair suffix.'''
    try:
        # Load the module dynamically based on the suffix
        module = __import__(f'src.regression.generated_scripts.summary{suffix}', fromlist=['regression_max_errors'])
        regression_max_errors = module.regression_max_errors
        return regression_max_errors
    except ImportError as e:
        pass
        # print(f"Error importing module for suffix {suffix}: {e}")

def get_repair_df(suffixes):
    '''Load the ensemble max errors for a given repair suffix.'''
    rows_regression = []
    rows_ensemble = []

    for suffix in suffixes:
        regression_max_errors = get_repair_regression_data(suffix)
        rowEnsemble = get_repair_ensemble_data(suffix)

        if not rowEnsemble or not regression_max_errors:
            continue 

        ensemble_max_error_majority, ensemble_max_error_best = rowEnsemble[3], rowEnsemble[4]
        rows_ensemble.append([suffix, ensemble_max_error_majority, ensemble_max_error_best])

        for k, values in enumerate(regression_max_errors):
            if k == 6:
                continue
            for count, value in enumerate(values):
                rows_regression.append([suffix, k, count+1, value])

    df_regression = pd.DataFrame(rows_regression, columns=['repair', 'k', 'comb', 'error'])
    df_ensemble = pd.DataFrame(rows_ensemble, columns=['repair', 'error_majority', 'error_best'])
    return df_regression, df_ensemble

####################
# Plotting
####################
def plot_regression_variance(suffixes, df_regression, df_ensemble):
    '''For each of the suffix, save a plot with suffix.pdf. The plot contains a box for df_regression having that suffix, and a line for df_ensemble'''

    for suffix in suffixes:
        if not suffix in df_regression['repair'].values:
            continue

        plt.figure(figsize=(8, 6.25))

        # Create a twin axis for the count of invalids
        ax = plt.gca()

        # Plot the count of invalids on the first y-axis

        # Plot the error percentages on the second y-axis
        df = df_ensemble[df_ensemble['repair'] == suffix]
        ax.axhline(y=df['error_best'].iloc[0], color='blue', linestyle=':', label='Minority Veto')
        ax.axhline(y=df['error_majority'].iloc[0], color='brown', linestyle='-', label='Majority Consensus')

        # sns.lineplot(x='repair', y='error_majority', data=df, ax=ax, marker='^', color='brown')
        # sns.lineplot(x='repair', y='error_best', data=df, ax=ax, marker='o', color='blue', dashes=(4, 4), linestyle=':')

        # Plot the regression errors
        df = df_regression[df_regression['repair'] == suffix]
        # Calculate mean and std dev for error bars
        means = df.groupby('k')['error'].mean()
        stds = df.groupby('k')['error'].std()
        
        # The x positions are the unique values of 'k'
        x_pos = means.index.values
        
        # The mean and error values
        means_vals = means.values
        yerrs = stds.values
        
        ax.errorbar(x=x_pos, y=means_vals, yerr=yerrs, fmt='-o', label='Regression @ s', color='black', capsize=5)

        ax.set_xlabel('Calibration Set Size (s)', fontsize=18)
        ax.set_ylabel('Maximum Absolute Error (%)', fontsize=18)
        ax.tick_params(axis='y', labelsize=12)

        # Add a legend with a grey box for the barplot
        # legend_elements = [
        #     plt.Line2D([0], [0], color='black', marker='x', label='Regression @ s=0', linestyle='-', markersize=10, ),

        #     plt.Line2D([0], [0], color='brown', marker='^', label='Majority Consensus'),
        #     plt.Line2D([0], [0], color='blue', marker='o', label='Minority Veto', linestyle=':'),

        #     plt.Line2D([0], [0], color='green', marker='s', label='Regression @ s=1', linestyle='--'),
        #     plt.Line2D([0], [0], color='red', marker='d', label='Regression @ s=3', linestyle='-.'),
        #     plt.Line2D([0], [0], color='purple', marker='*', label='Regression @ s=5', dashes=(2, 2), linestyle='--', markersize=10),
        #     Patch(facecolor='lightgray', label='Missing Values')
        # ]
        # ax.legend(handles=legend_elements, fontsize=16, loc='upper center', bbox_to_anchor=(0.5, 1.45), ncol=2)

        # Set the x-ticks to be the repair configurations
        # xticks = ['+ ' + i if i != 'Original' else i for i in df['repair'].tolist()]
        # ax.set_xticks(range(len(xticks)))
        ax.tick_params(axis='x', labelsize=14)
        ax.set_ylim(0, 18)

        # Ensure both axes have the same y-axis ticks
        ax.tick_params(axis='y', labelsize=14)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.set_yticks(ax.get_yticks())

        # Add a legend
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], fontsize=16, loc='upper center', bbox_to_anchor=(0.5, 1.45))


        plt.tight_layout()
        plt.savefig(f'{pathRegression}/regression_variance{suffix}.pdf', bbox_inches='tight')

####################
# Main
####################

def regression_variance():
    '''Plot the repair vs ensemble errors.'''

    # Get the suffixes for the repair configurations
    suffixes = get_repair_suffixes()

    # Load the DF for each suffix
    df_regression, df_ensemble = get_repair_df(suffixes)

    # Plot the data
    plot_regression_variance(suffixes, df_regression, df_ensemble)