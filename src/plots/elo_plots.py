import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import ELO_RATING, MODELS_VAL, MODELS_SHORT, pathImages
from src.validation.generated_scripts.llm_as_judge_fcnhp import ensemble_predicted_precisions, validator_tnr

def elo_vs_precision():
    """
    Plot Elo vs Precision for the ensemble.
    """

    plt.figure(figsize=(8, 4))

    rows = [(modelName, ensemble_predicted_precisions[i]*100, ELO_RATING[modelName]) for i, modelName in enumerate(MODELS_VAL)]

    df_plot_rest = pd.DataFrame(rows, columns=['generator', 'precision', 'elo'])
    # df_plot_human = pd.DataFrame(rows, columns=['generator', 'precision'])

    sns.scatterplot(data=df_plot_rest, x='elo', y='precision', hue='generator', palette='deep', s=150)
    # sns.scatterplot(data=df_plot_human, x='elo', y='precision', hue='generator', palette='deep')
    # plt.title('Elo Rating vs Precision')
        
    # Add annotations for each point
    rows = list(df_plot_rest.iterrows()) + list(df_plot_rest.iterrows())
    for idx, row in rows:
        shorthand = MODELS_SHORT.get(row['generator'], row['generator'])
        # Use a star symbol for models in MODELS_GEN to indicate "ground truth"
        # if row['generator'] in df_pg['generator'].values:
        #     plt.scatter(row['elo'], row['precision'], color='gold', marker='*', s=150, zorder=4, label='Ground Truth' if idx == 0 else "")
        
        elo = row['elo']
        prec = row['precision']


        offset = (0, -20)
        if shorthand == 'GPT 4T':
            shorthand = 'GPT 4'
        elif shorthand == 'Gemini 2.5-P':
            offset = (-90, -5)
        elif shorthand == 'DeepSeek 2.5':
            offset = (-20, -30)
        elif shorthand == 'Qwen Coder-P':
            offset = (-10, -30)
        elif shorthand == 'Sonnet 3.5':
            offset = (-70, 30)
        elif shorthand == 'Opus 3':
            offset = (-60, 20)
        elif shorthand == 'Gemini 1.5-F':
            offset = (0, -25)
        elif shorthand == 'Gemini 1.5-P':
            offset = (20, -15)
        plt.annotate(shorthand,  # Use shorthand for model name for clarity
                xy=(row['elo'], row['precision']),
                xytext=offset,  # Offset text slightly to the right
                textcoords='offset points',
                fontsize=12,
                alpha=0.8,
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.1', alpha=0.6))

    plt.ylim(84, 100)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # plt.title('Elo Rating vs Precision', fontsize=16)
    plt.xlabel('Elo Rating', fontsize=14)
    plt.ylabel('Precision %', fontsize=14)
    plt.legend().remove()  # Remove the legend since we have annotations
    plt.tight_layout()
    plt.savefig(f'{pathImages}/others/elo_precision.pdf')


def elo_vs_tnr():
    df_plot = pd.DataFrame({
        'elo': [ELO_RATING[model] for model in MODELS_VAL],
        'tnr': [validator_tnr[i] * 100 for i in range(len(MODELS_VAL))],
        'validator': MODELS_VAL
    })

    plt.figure(figsize=(8, 4))
    # Create the scatterplot without jitter to keep actual data points accurate
    sns.scatterplot(
        x='elo', 
        y='tnr', 
        data=df_plot,
        hue='validator', 
        palette='deep',
        s=150,
        alpha=0.8
    )

    plt.xlabel('Elo Rating', fontsize=14)
    plt.ylabel('TNR (True Negative Rate) %', fontsize=14)

    # Improved annotation positioning to avoid overlaps in the 10-30 TNR range and 1200-1300 Elo range
    model_offsets = {
        'gpt-4-turbo': (-30, 15),       # Move up and right
        'gpt-4o-mini': (15, 15),      # Move down and right
        'gpt-4o': (35, 0),          # Move down and left
        'claude_3_opus': (-60, 5),    # Move up and left
        'claude_3.5_sonnet': (15, -5),   # Move right
        'gemini-1.5-flash': (10, -5), # Move down and right
        'gemini-1.5-pro': (-10, -20),  # Move down and left
        'qwen-coder-plus': (-15, 20),  # Move up and left
        'deepseek-chat': (-15, -30),   # Move down and left
        'claude_3.5_haiku': (10, -15), # Move down and right
        'gemini-2.5-flash': (-35, -25), # Keep default
        'gemini-2.5-pro': (-55, -25),   # Keep default
        'gpt-4.1-2025-04-14': (10, -15),  # Move up and left
        'gpt-4.1-mini': (15, -10), # Move up and right
        'gemini-1.5-flash': (-30, -40),  # Move down
    }

    # Add annotations with custom positioning for each model
    for idx, row in df_plot.iterrows():
        shorthand = MODELS_SHORT.get(row['validator'], row['validator'])
        
        # Get the custom offset for this model
        x_offset, y_offset = model_offsets.get(row['validator'], (15, 15))
        
        plt.annotate(
            shorthand,
            xy=(row['elo'], row['tnr']),  # Point to exact data point
            xytext=(x_offset, y_offset),  # Use custom offset
            textcoords='offset points',
            fontsize=12,
            alpha=0.8,
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.1', alpha=0.6)
        )

    plt.ylim(0, 55)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend().remove()  # Remove the legend since we have annotations
    plt.tight_layout()
    plt.savefig(f'{pathImages}/others/elo_tnr.pdf')