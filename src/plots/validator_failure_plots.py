from src.config import Model, pathImages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import MODELS_SHORT, MODELS_SHORT_ORDERED_RELEASE, MODELS_GEN, MODELS_VAL, VALIDATOR_REPAIR_SUFFIX, fpathEnsembleResultsNoSuffix, ValidatorRepairConfig

pathValidator = f'{pathImages}/validator'


####################
# Validation failures
####################
def plot_failure_counts(error_message_counts_validator):
    """
    Plot the counts of validation failures for each model.
    
    Args:
        error_message_counts_validator (dict): Dictionary containing {validator_model: {reason_failure: count}}.
    """
    # Error short hand
    error_shorthand = {
        # "missing_generator_line_number": "Others",
        "missing_output": "Incorrect Output",
        "missing_label": "Missing Label",
        "missing_line_number": "Missing Line Number",
        "unmatched_feedback": "Missing Feedback"
    }
    
    # Convert the nested dictionary into a DataFrame
    data = []
    for model, errors in error_message_counts_validator.items():
        for error, count in errors.items():
            error = error_shorthand.get(error, error)  # Use shorthand for the error message
            model = MODELS_SHORT.get(model, model)  # Use shorthand for the model name
            data.append({'model': model, 'error': error, 'count': count})

    df_errors = pd.DataFrame(data)
    
    # Pivot the DataFrame for stacked bar plot
    df_pivot = df_errors.pivot(index='model', columns='error', values='count').fillna(0)
    
    # Ensure the order of error types matches error_shorthand.values()
    error_order = list(error_shorthand.values())
    df_pivot = df_pivot.reindex(columns=[col for col in error_order if col in df_pivot.columns])

    # Reorder the DataFrame based on MODELS_SHORT_ORDERED_RELEASE
    df_pivot = df_pivot.reindex(index=[model for model in MODELS_SHORT_ORDERED_RELEASE if model in df_pivot.index])
    
    # Stacked bar plot
    plt.figure(figsize=(6.5, 3.5))
    df_pivot.plot(kind='bar', stacked=True)
    # plt.yscale('log')
    plt.xlabel('LLM Validator', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.ylim(0, 2500)  # Set y-axis limit to 2500
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[::-1], labels[::-1], bbox_to_anchor=(0.5, 1.05), loc='lower center', ncol=2, fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{pathValidator}/failure_counts{VALIDATOR_REPAIR_SUFFIX}.pdf', bbox_inches='tight')

####################
# Counting plot
####################
def get_df_ensemble_merged(suffix):
    fpathEnsembleResults = f'{fpathEnsembleResultsNoSuffix}{suffix}.xlsx'
    df_merged = pd.DataFrame()

    for model in MODELS_GEN:
        df = pd.read_excel(fpathEnsembleResults, sheet_name=model)
        df['generator'] = model  # Add a column for the model name
        df_merged = pd.concat([df_merged, df], ignore_index=True)

    return df_merged


def count_validator(df, label):
    '''Count the number of validator models that got the valid label right'''
    df_filter = df[df['y'] == label]  # Filter for valid labels
    cols = [f'y_{model}' for model in MODELS_VAL] # Columns for validator models

    # If label is 1, sum the columns; if label is 0, count the number of zeros
    if label == 1:
        series_sum = df_filter[cols].sum(axis=1)
    else:
        series_sum = df_filter[cols].apply(lambda x: (x == 0).sum(), axis=1)

    # Convert this into a dictionary {1: count, 2: count, ...}
    count_dict = series_sum.value_counts().to_dict()

    # Convert count dicts to percentage of total
    num_labels = sum(count_dict.values())
    # If there are no labels, return an empty DataFrame
    if num_labels == 0:
        return pd.DataFrame(columns=['count', 'label_percentage', 'cumulative_percentage'])
    count_dict = {k: v / num_labels * 100 for k, v in count_dict.items()}

    # Convert to DF
    df_count_dict = pd.DataFrame(list(count_dict.items()), columns=['count', 'label_percentage'])
    df_count_dict['count'] = df_count_dict['count'].astype(int)
    df_count_dict = df_count_dict.sort_values(by='count', ascending=False).reset_index(drop=True)

    # Add a cumulative percentage column
    df_count_dict['cumulative_percentage'] = df_count_dict['label_percentage'].cumsum()

    return df_count_dict


def plot_validator_valid():
    '''Plot the number of validator models that got the valid label right'''
    suffixTrue = ValidatorRepairConfig(default=True).getSuffix()
    suffixFalse = ValidatorRepairConfig(default=False).getSuffix()

    df_repair = get_df_ensemble_merged(suffixTrue)
    df_noRepair = get_df_ensemble_merged(suffixFalse)

    df_repair_valid = count_validator(df_repair, 1)
    df_noRepair_valid = count_validator(df_noRepair, 1)

    # Combine the dataframes for side-by-side bar plots
    df_noRepair_valid['type'] = 'Original'
    df_repair_valid['type'] = 'Repaired'
    df_combined = pd.concat([df_noRepair_valid, df_repair_valid])

    # Plot
    plt.figure(figsize=(8, 4))

    # Create side-by-side bar plots
    sns.barplot(x='count', y='cumulative_percentage', hue='type', data=df_combined, alpha=0.8)
    plt.xlabel('Number of Validator Models that Got Valid Label Right', fontsize=16)
    plt.ylabel('Percentage of Valid Labels', fontsize=16)

    # Font size adjustments
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    
    plt.savefig(f'{pathValidator}/validator_validCount.pdf', bbox_inches='tight')

def plot_validator_invalid():
    '''Plot the number of validator models that got the invalid label right'''
    suffixTrue = ValidatorRepairConfig(default=True).getSuffix()
    suffixFalse = ValidatorRepairConfig(default=False).getSuffix()

    df_repair = get_df_ensemble_merged(suffixTrue)
    df_noRepair = get_df_ensemble_merged(suffixFalse)

    # Combine the dataframes for side-by-side bar plots
    df_repair_invalid = count_validator(df_repair, 0)
    df_noRepair_invalid = count_validator(df_noRepair, 0)

    # Combine the dataframes for side-by-side bar plots
    df_noRepair_invalid['type'] = 'Original'
    df_repair_invalid['type'] = 'Repaired'
    df_combined_invalid = pd.concat([df_noRepair_invalid, df_repair_invalid])

    # Plot
    plt.figure(figsize=(8, 4))
    sns.barplot(x='count', y='label_percentage', hue='type', data=df_combined_invalid, alpha=0.8)
    plt.xlabel('Number of Validator Models that Got Invalid Label Right', fontsize=14)
    plt.ylabel('Percentage of Invalid Labels', fontsize=14)
    plt.legend()

    # Font size adjustments
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)

    plt.savefig(f'{pathValidator}/validator_invalidCount.pdf', bbox_inches='tight')

def plot_validator_valid_invalid_cumsum():
    '''Plot the cumulative sum of valid and invalid labels for validator models.'''
    suffixTrue = ValidatorRepairConfig(default=True).getSuffix()
    suffixFalse = ValidatorRepairConfig(default=False).getSuffix()

    df_repair = get_df_ensemble_merged(suffixTrue)

    # Count valid and invalid labels
    df_repair_valid = count_validator(df_repair, 1)
    df_repair_invalid = count_validator(df_repair, 0)

    # Combine the dataframes for side-by-side bar plots
    df_repair_valid['type'] = 'Valid Label (Repaired)'
    df_repair_invalid['type'] = 'Invalid Label (Repaired)'

    df_combined_cumsum = pd.concat([df_repair_valid, df_repair_invalid])

    # Plot
    plt.figure(figsize=(8, 4))
    sns.lineplot(
        x='count', 
        y='cumulative_percentage', 
        hue='type', 
        data=df_combined_cumsum, 
        alpha=0.8, 
        marker='o', 
        palette={'Valid Label (Repaired)': 'blue', 'Invalid Label (Repaired)': 'red'}
    )
    plt.gca().invert_xaxis()

    plt.xlabel('Minimum Number of Validators that Got Label Right', fontsize=14)
    plt.ylabel('Cumulative Number of Labels (%)', fontsize=14)
    
    # Font size adjustments
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.savefig(f'{pathValidator}/validator_valid_invalidCount.pdf', bbox_inches='tight')