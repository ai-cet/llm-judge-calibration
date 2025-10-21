from src.config import Model, pathImages, pathLogs
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import pGa_CONST, MODELS_SHORT, MODELS_SHORT_ORDERED_RELEASE, MODELS_GEN, MODELS_VAL, VALIDATOR_REPAIR_SUFFIX
import pickle

pathValidator = f'{pathImages}/validator'

# Load the pickle file containing GV data
with open(f'{pathLogs}/all_logs.pkl', 'rb') as file:
    Gis = pickle.load(file)

####################
# GV Plot
####################
def get_df_regression(df_pg):
    ga = []
    for model in MODELS_VAL:
        filtered = df_pg[df_pg['generator'] == MODELS_SHORT[model]]
        ga.append(filtered['precision'].iat[0]/100 if not filtered.empty else None)

    rows = []
    for k, karr in enumerate(Gis):
        if k not in [0,1,3,5]:
            continue
        maxErr = float('-inf')
        maxCombIndex = None
        maxComb = None

        for comb, arr in enumerate(karr):
            ghat = arr[2]
            err = 0
            for hat, actual in zip(ghat, ga):
                if actual is not None:
                    err += (hat - actual)
            err /= len(ga)

            if err > maxErr:
                maxErr = err
                maxCombIndex = comb
                maxComb = ghat

        for model, prec in zip(MODELS_VAL, maxComb):
            model = MODELS_SHORT[model]
            rows.append((k, maxCombIndex, model, prec*100, maxErr*100))

    df_regression = pd.DataFrame(rows, columns=['k', 'comb', 'generator', 'precision', 'maxErr'])
    return df_regression

def get_df_pg():
    rows = [(MODELS_SHORT[model], pGa_CONST[model] * 100, 'human') for model in pGa_CONST.keys()]
    df_pg = pd.DataFrame(rows, columns=['generator', 'precision', 'validator'])
    return df_pg
    
def regression_plot():
    df_pg = get_df_pg()
    df_regression = get_df_regression(df_pg)

    # Create a figure
    plt.figure(figsize=(8, 6))

    # Group by generator to get statistics for each
    grouped = df_regression.groupby('generator')
    stats = grouped.agg({'precision': ['mean', 'min', 'max']})
    stats.columns = ['mean', 'min', 'max']
    stats = stats.reset_index()

    # Sort by mean precision for better visualization
    # stats = stats.sort_values('mean', ascending=True)

    # Sort the stats dataframe based on MODELS_SHORT_ORDERED_RELEASE
    stats['generator'] = stats['generator'].apply(lambda x: MODELS_SHORT.get(x, x))
    stats = stats.sort_values(by='generator', 
                                key=lambda x: [MODELS_SHORT_ORDERED_RELEASE.index(model) if model in MODELS_SHORT_ORDERED_RELEASE else float('inf') for model in x])


    # Add a scatter for human ground truth evaluations from df_pg
    for i, row in df_pg.iterrows():
        # Find the position in stats dataframe
        gen_idx = stats[stats['generator'] == row['generator']].index
        if len(gen_idx) > 0:
            idx = stats.index.get_loc(gen_idx[0])
            plt.scatter([idx], [row['precision']], color='black', marker='o', s=50, zorder=4, alpha=0.8,
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
    
    # Add scatter for each k
    # Add a line plot for each k value
    for k in df_regression['k'].unique(): 
        k_data = df_regression[df_regression['k'] == k]
        # Sort k_data by generator order to match the x-axis
        k_data = k_data.sort_values(by='generator', 
                                   key=lambda x: [MODELS_SHORT_ORDERED_RELEASE.index(model) if model in MODELS_SHORT_ORDERED_RELEASE else float('inf') for model in x])
        
        print(k_data)
        
        # Map generator names to x positions
        # x_positions = [stats[stats['generator'] == gen].index[0] for gen in k_data['generator']]
        
        plt.plot(range(len(k_data)), k_data['precision'], marker='o', linewidth=2, markersize=4, 
                 alpha=0.7, label=f'k={k}')

    # Set x-tick labels to model names, using shorthand for readability
    plt.xticks(range(len(stats)), [MODELS_SHORT.get(model, model) for model in stats['generator']], 
            rotation=45, ha='right', fontsize=14)

    plt.ylabel('Precision', fontsize=16)
    plt.xlabel('Generator', fontsize=16)
    plt.grid(True, axis='y', alpha=0.3)
    # plt.title('Generator Precision by Different Validators', fontsize=18, pad=15)
    plt.legend(fontsize=14, loc='upper center', bbox_to_anchor=(0.5, 1.35), ncols=3)

    # Increase the size of tick labels
    plt.yticks(fontsize=14)

    # Save with high DPI for crisp text
    plt.tight_layout()
    plt.savefig(f'{pathImages}/regression/generator_precision.pdf', bbox_inches='tight')