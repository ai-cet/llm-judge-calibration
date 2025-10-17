from src.config import Model, pathImages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import pGa_CONST, MODELS_SHORT, MODELS_SHORT_ORDERED_RELEASE, MODELS_GEN, MODELS_VAL, VALIDATOR_REPAIR_SUFFIX

pathValidator = f'{pathImages}/validator'

####################
# GV Plot
####################
def get_df_gv(GV):
    # Create the DF
    li = []
    for i, model1 in enumerate(MODELS_VAL):
        for j, model2 in enumerate(MODELS_VAL):
            generator = MODELS_SHORT[model1]
            validator = MODELS_SHORT[model2]
            li.append([generator, validator, GV[i][j]*100])
    df_gv = pd.DataFrame(li, columns=['generator', 'validator', 'precision'])
    
    return df_gv

def get_df_pg():
    rows = [(MODELS_SHORT[model], pGa_CONST[model] * 100, 'human') for model in pGa_CONST.keys()]
    df_pg = pd.DataFrame(rows, columns=['generator', 'precision', 'validator'])
    return df_pg

def gv_plot(GV):
    df_pg = get_df_pg()
    df_gv = get_df_gv(GV)

    # Create a figure
    plt.figure(figsize=(12, 4))

    # Group by generator to get statistics for each
    grouped = df_gv.groupby('generator')
    stats = grouped.agg({'precision': ['mean', 'min', 'max']})
    stats.columns = ['mean', 'min', 'max']
    stats = stats.reset_index()

    # Sort by mean precision for better visualization
    # stats = stats.sort_values('mean', ascending=True)

    # Sort the stats dataframe based on MODELS_SHORT_ORDERED_RELEASE
    stats['generator'] = stats['generator'].apply(lambda x: MODELS_SHORT.get(x, x))
    stats = stats.sort_values(by='generator', 
                                key=lambda x: [MODELS_SHORT_ORDERED_RELEASE.index(model) if model in MODELS_SHORT_ORDERED_RELEASE else float('inf') for model in x])

    # Plot vertical lines from min to max
    for i, row in enumerate(stats.itertuples()):
        plt.plot([i, i], [row.min, row.max], 'k-', linewidth=1.5, color='gray', alpha=0.75)
        
    # Plot max and mean markers
    plt.scatter(range(len(stats)), stats['max'], color='green', s=70, zorder=3, marker='^', label='Validator Max')
    plt.scatter(range(len(stats)), stats['mean'], color='black', s=200, zorder=3, marker='_', label='Validator Mean')

    # Add a scatter for human ground truth evaluations from df_pg
    for i, row in df_pg.iterrows():
        # Find the position in stats dataframe
        gen_idx = stats[stats['generator'] == row['generator']].index
        if len(gen_idx) > 0:
            idx = stats.index.get_loc(gen_idx[0])
            plt.scatter([idx], [row['precision']], color='blue', marker='o', s=50, zorder=4, alpha=0.8,
                        label='Ground Truth' if i == 0 else "")
            
            # Add annotation for the human evaluation with precision multiplied by 100
            # plt.annotate(f"{row['precision']:.1f}", 
            #             xy=(idx, row['precision']), 
            #             xytext=(20, -15),  # Offset below the point
            #             textcoords='offset points', 
            #             ha='center', 
            #             color='darkgoldenrod',
            #             fontweight='bold',
            #             fontsize=10)
    
    # Add scatter for min precision (after Gold)
    plt.scatter(range(len(stats)), stats['min'], color='red', s=70, zorder=3, marker='v', label='Validator Min')

    # Set x-tick labels to model names, using shorthand for readability
    plt.xticks(range(len(stats)), [MODELS_SHORT.get(model, model) for model in stats['generator']], 
            rotation=30, ha='right', fontsize=14)
    plt.yticks(fontsize=14)

    plt.ylabel('Precision', fontsize=16)
    plt.xlabel('Generator', fontsize=16)
    plt.grid(True, axis='y', alpha=0.3)
    # plt.title('Generator Precision by Different Validators', fontsize=18, pad=15)
    # plt.legend(fontsize=16, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncols=4)
    plt.legend(fontsize=14, loc='center left', bbox_to_anchor=(1, 0.5))

    

    # Increase the size of tick labels
    plt.yticks(fontsize=16)

    # Save with high DPI for crisp text
    # plt.tight_layout()
    plt.savefig(f'{pathValidator}/gv{VALIDATOR_REPAIR_SUFFIX}.pdf', bbox_inches='tight')

def gv_boxplot(GV):
    df_pg = get_df_pg()
    df_gv = get_df_gv(GV)

    # Create a figure
    plt.figure(figsize=(14, 5))

    # Create box plot data grouped by generator
    # Sort generators based on MODELS_SHORT_ORDERED_RELEASE
    df_gv['generator'] = df_gv['generator'].apply(lambda x: MODELS_SHORT.get(x, x))
    df_gv = df_gv.sort_values(by='generator', 
                             key=lambda x: [MODELS_SHORT_ORDERED_RELEASE.index(model) if model in MODELS_SHORT_ORDERED_RELEASE else float('inf') for model in x])
    
    # Create box plot
    sns.boxplot(data=df_gv, x='generator', y='precision', fill=False, linecolor='black')
    
    # Get stats for ground truth positioning
    stats = df_gv.groupby('generator').agg({'precision': ['mean']}).reset_index()
    stats.columns = ['generator', 'mean']

    # Add a scatter for human ground truth evaluations from df_pg
    for i, row in df_pg.iterrows():
        # Find the position in stats dataframe
        gen_idx = stats[stats['generator'] == row['generator']].index
        if len(gen_idx) > 0:
            idx = stats.index.get_loc(gen_idx[0])
            plt.scatter([idx], [row['precision']], color='gold', marker='*', s=250, zorder=4, 
                        label='Ground Truth' if i == 0 else "")
            
            # Add annotation for the human evaluation with precision multiplied by 100
            plt.annotate(f"{row['precision']:.1f}", 
                        xy=(idx, row['precision']), 
                        xytext=(20, -15),  # Offset below the point
                        textcoords='offset points', 
                        ha='center', 
                        color='darkgoldenrod',
                        fontweight='bold',
                        fontsize=12)


    # Set x-tick labels to model names, using shorthand for readability
    plt.xticks(rotation=45, ha='right', fontsize=13)

    plt.ylabel('Precision', fontsize=16)
    plt.xlabel('Generator', fontsize=16)
    plt.grid(True, axis='y', alpha=0.3)
    # plt.title('Generator Precision by Different Validators', fontsize=18, pad=15)
    plt.legend(fontsize=13)
    plt.tight_layout()

    # Increase the size of tick labels
    plt.yticks(fontsize=13)

    # Save with high DPI for crisp text
    plt.savefig(f'{pathValidator}/gv_box{VALIDATOR_REPAIR_SUFFIX}.pdf', bbox_inches='tight')

####################
# True Positive Rate vs True Negative Rate Plot
####################
def get_df_validator_tpr_tnr(validator_tpr, validator_tnr):
    # Create the DF
    df_validator = pd.DataFrame({
        'validator': [model for model in MODELS_VAL],
        'tpr': [tpr * 100 for tpr in validator_tpr],
        'tnr': [tnr * 100 for tnr in validator_tnr]
    })
            
    return df_validator

def add_offsets(df_validator):
    # Improved annotation positioning to avoid overlaps in the 10-30 TNR range and 1200-1300 Elo range
    model_offsets = {
        MODELS_SHORT[Model.GPT_4_TURBO.value]: (-30, 10),       # Move up and right
        MODELS_SHORT[Model.GEMINI_1_5_FLASH.value]: (-90, -5), # Move down and right

        MODELS_SHORT[Model.GPT_4O_MINI.value]: (10, -5),      # Move down and right
        MODELS_SHORT[Model.GPT_4O.value]: (30, 10),          # Move down and left
        MODELS_SHORT[Model.CLAUDE_3_OPUS.value]: (-20, -65),    # Move up and left
        MODELS_SHORT[Model.CLAUDE_3_5_SONNET.value]: (40, -65),   # Move right

        MODELS_SHORT[Model.GEMINI_1_5_PRO.value]: (-10, -30),  # Move down and left
        MODELS_SHORT[Model.QWEN_CODER_PLUS.value]: (-90, -25),  # Move up and left
        MODELS_SHORT[Model.DEEPSEEK_CHAT.value]: (-45, -60),   # Move down and left
        MODELS_SHORT[Model.CLAUDE_3_5_HAIKU.value]: (-45, -30), # Move down and right
        MODELS_SHORT[Model.GPT_4_1.value]: (-256, -15),  # Move up and left
        MODELS_SHORT[Model.GPT_4_1_MINI.value]: (-250, -10), # Move up and right

        MODELS_SHORT[Model.GEMINI_2_5_FLASH.value]: (-25, -25), # Keep default
        MODELS_SHORT[Model.GEMINI_2_5_PRO.value]: (-25, -25),   # Keep default
    }

    # Add annotations for each point
    for idx, row in df_validator.iterrows():
        shorthand = MODELS_SHORT.get(row['validator'], row['validator'])
        x_offset, y_offset = model_offsets.get(row['validator'], (0, 0))
        
        if shorthand in [MODELS_SHORT[Model.GEMINI_1_5_FLASH.value], 
                        MODELS_SHORT[Model.GEMINI_2_5_PRO.value], 
                        MODELS_SHORT[Model.GEMINI_2_5_FLASH.value]]: 
            
            # Add arrow pointing to the marker
            plt.annotate(shorthand,
                            xy=(row['tpr'], row['tnr']), 
                            xytext=(x_offset, y_offset),
                            textcoords='offset points',
                            fontsize=12,
                            alpha=0.8,
                            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.1', alpha=0.6))

def tpr_tnr_validator(validator_tpr, validator_tnr, ensemble_tpr, ensemble_tnr):
    plt.figure(figsize=(12, 3))

    df_validator = get_df_validator_tpr_tnr(validator_tpr, validator_tnr)

    # Replace the validator names with shorthand names
    df_validator['validator'] = df_validator['validator'].apply(lambda x: MODELS_SHORT.get(x, x))

    # Sort the df based on reverse order of release date
    df_validator = df_validator.sort_values(by='validator', ascending=False, 
                                            key=lambda x: [MODELS_SHORT_ORDERED_RELEASE.index(model) if model in MODELS_SHORT_ORDERED_RELEASE else float('inf') for model in x])

    # Scatter plot for TPR vs TNR
    sns.scatterplot(data=df_validator, x='tpr', y='tnr', hue='validator', style='validator', palette='deep', s=100)

    # Add offsets to the annotations
    add_offsets(df_validator)

    # Use shorthand names for the legend instead of full model names
    handles, labels = plt.gca().get_legend_handles_labels()
    # Sort the handles and labels based on the order of MODELS_SHORT_ORDERED_RELEASE
    # sorted_labels = [MODELS_SHORT[model] for model in MODELS_SHORT_ORDERED_RELEASE if model in MODELS_SHORT]
    # sorted_handles = [handles[labels.index(label)] for label in sorted_labels if label in labels]
    # plt.legend(handles, labels, bbox_to_anchor=(0.5, 1.05), loc='lower center', ncol=4, fontsize=12)
    plt.legend(handles, labels, bbox_to_anchor=(1.05, 0.5), loc='center left', ncol=2, fontsize=14)


    plt.ylim(0,60)
    plt.xlim(82,100)

    # Labels and fontsize
    plt.xlabel('True Positive Rate %', fontsize=14)
    plt.ylabel('True Negative Rate %', fontsize=14)

    # Add dotted line for the ensemble
    rows_valid = []
    rows_invalid = []
    for i in range(1, len(MODELS_VAL)+1):
        modelName = f'v{i}'
        rows_valid.append({
            'validator': 'valid-voting',
            'tpr': ensemble_tpr[modelName] * 100,
            'tnr': ensemble_tnr[modelName] * 100
        })

        modelName = f'i{i}'
        rows_invalid.append({
            'validator': 'invalid-voting',
            'tpr': ensemble_tpr[modelName] * 100,
            'tnr': ensemble_tnr[modelName] * 100
        })
    df_valid = pd.DataFrame(rows_valid )
    df_invalid = pd.DataFrame(rows_invalid)

    # Scatter plot with connected lines for valid and invalid voting
    # sns.lineplot(data=df_valid, x='tpr', y='tnr', marker='o', label='Valid Voting', color='blue', alpha=0.8)
    # sns.lineplot(data=df_invalid, x='tpr', y='tnr', marker='o', label='Invalid Voting', color='red', alpha=0.8)

    plt.grid(True, linestyle='--', alpha=0.5)

    # plt.axhline(y=25, color='gray', linestyle='--', alpha=0.2)
    # plt.axvline(x=94, color='gray', linestyle='--', alpha=0.2)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.tight_layout()
    plt.savefig(f'{pathValidator}/tpr_tnr_validator{VALIDATOR_REPAIR_SUFFIX}.pdf', bbox_inches='tight')

