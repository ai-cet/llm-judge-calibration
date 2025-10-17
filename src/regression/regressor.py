import numpy as np
from scipy.optimize import minimize
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.regression.config import COLOR_GREEN_DELTA, COLOR_YELLOW_DELTA, COMPUTE_REFERENCE_VALUES, ENSEMBLE_MAJORITY_MAXERR, ENSEMBLE_MAJORITY_MEANERR, ENSEMBLE_BEST_MAXERR, ENSEMBLE_BEST_MEANERR, K_LIST, LLM_VALIDATOR_MAXERR_RANGE, LLM_VALIDATOR_MEANERR_RANGE, LOSS_PRED, NUM_RUNS, MAX_WORKERS, WEIGHTS
from src.regression.config import PV_START, PVIV_START, PG_START, ERR_EPSILON_PG, ERR_EPSILON_PIV, ERR_EPSILON_PV, GENS, MODELS, MODEL_NAMES, MODEL_ENUM
from src.regression.config import VALIDATOR_COUNTS_CONST, pGa_CONST, GV_CONST, PVVA_CONST, PVIVA_CONST
from src.regression.config import PATH_LOGS, PATH_LATEX, PATH_REGRESSION, PATH_LOGS_LOSS_ITER, fpathRegressionSummary
from src.config import VALIDATOR_REPAIR_SUFFIX

from src.regression.loss import total_loss, write_loss_to_csv

import sys

from src.validation.utils import pretty_print_into_file
np.random.seed(42)

import warnings
import os
warnings.filterwarnings("always", category=UserWarning)

####################
# Setup
####################

# Create necessary directories if they do not exist
if not os.path.exists(PATH_REGRESSION):
    os.makedirs(PATH_REGRESSION)

msg = '# AUTO-GENERATED FILE. DO NOT EDIT\n'
with open(fpathRegressionSummary, 'w') as f:
    f.write(msg)  # Clear the file

# =========================
#   READ DATA
# =========================

def read_data():
    GV = GV_CONST
    pGa = pGa_CONST
    VALIDATOR_COUNTS = VALIDATOR_COUNTS_CONST
    PVVA = PVVA_CONST
    PVIVA = PVIVA_CONST

    return GV, pGa, VALIDATOR_COUNTS, PVVA, PVIVA
        


# =========================
#   PRETTY PRINT
# =========================

# Initialize the file to empty
if not os.path.exists(PATH_LATEX):
    os.makedirs(PATH_LATEX)
with open(f"{PATH_LATEX}GV_hat.tex", "w") as f:
    f.write('')

def write_latex_table(k1, pVv, pViv, pG_min, pG_max, pG_mean, PVVA, PVIVA, pGa):
    G_hat = np.outer(pG_mean, pVv) + np.outer((1 - pG_mean), (1 - pViv))
    
    s = f'\n\nk = {k1}'

    s += r'''
  \begin{adjustbox}{max width=\textwidth}
     \begin{tabular}{@{}ccccccccccc||cccc}
        \toprule
        &  & \multicolumn{10}{c}{\textbf{Validators}} \\
        \cmidrule(l){2-15}
         \textbf{Generators} & \textbf{3.5-turbo} & \textbf{4-turbo} & \textbf{4o-mini} & \textbf{4o} & \textbf{3 opus} & \textbf{3.5 sonnet} & \textbf{1.5 flash} & \textbf{1.5 pro} & \textbf{qwen} & \textbf{deepseek} & \textbf{$\hat{\pgen}$} & \textbf{$\max(\hat{\pgen})$} & \textbf{$\min(\hat{\pgen})$} & GT \\
    \midrule
    '''

    for i, row in enumerate(G_hat):
        s += f'\\textbf{{{MODEL_NAMES[i]}}} & '

        for val in row:
            s += f"{(100*val):.1f}\\% & "

        s += f"{(100*pG_mean[i]):.1f}\\% & "
        s += f"{(100*pG_max[i]):.1f}\\% & "
        s += f"{(100*pG_min[i]):.1f}\\% & "
        if pGa.get(MODELS[i], None) is not None:
            s += f"{(100*pGa[MODELS[i]]):.1f}\\% \\\\ \n"
        else:
            s += f"\\\\ \n"

    s += r'\midrule' + '\n'

    s += r'$\hat{\pvalid}$ '
    for x in pVv:
        s += f'& {100*x:.1f} '

    s += '\\\\ \n'

    s += r'$\pvalid$ '
    for x in PVVA:
        s += f'& {100*x:.1f} '

    s += '\\\\ \n'

    s += r'$\hat{\pinvalid}$ '
    for x in pViv:
        s += f'& {100*x:.1f} '

    s += '\\\\ \n'

    s += r'$\pinvalid$ '
    for x in PVIVA:
        s += f'& {100*x:.1f} '

    s += '\\\\ \n'
    s += r'''\bottomrule
    \end{tabular}%
  \end{adjustbox}
\end{table*}'''

    with open(f"{PATH_LATEX}GV_hat.tex", "a") as f:
        f.write(s)

def print_values_pair(Y, Y_HAT, labels):
    for y, y_hat, label in zip(Y, Y_HAT, labels):
        diff = abs(y - y_hat)
        color = "\033[92m" if diff < COLOR_GREEN_DELTA \
            else "\033[93m" if diff < COLOR_YELLOW_DELTA \
            else "\033[91m"
        print(f"{label}: {y*100:.1f}% vs {y_hat*100:.1f}% {color}(Δ{diff*100:.1f}%)\033[0m", end='  ')


def print_values_k(k1, numComb, j, pGa, VALIDATOR_COUNTS, log, max_errors, avg_errors, PVVA, PVIVA):
    print(f"\n=========================")
    print(f"k{k1} C{numComb} with {j}")
    print(f"=========================")

    G, Ghat, errorG, pVv, pViv = extract_values_log(pGa, VALIDATOR_COUNTS, log)
    GV_hat = np.outer(Ghat, pVv) + np.outer((1 - Ghat), (1 - pViv))
    
    print(f'\tG vs G_hat:', end=' ')
    print_values_pair(G, Ghat, pGa.keys())

    print(f"\n\tV+ vs V+_hat:", end=' ')
    print_values_pair(PVVA, pVv, MODELS)

    print(f"\n\tV- vs V-_hat:", end=' ')
    print_values_pair(PVIVA, pViv, MODELS)
    
    print(f"\n\tMax G Error (Validation): {max_errors[-1]}")
    print(f"\tAvg G Error (Validation): {avg_errors[-1]}")

    print(GV_hat)

    print(f"\tG Error (Test): {errorG}")

# =========================
#   EXTRACT VALUES
# ==========================

def extract_values_log(pGa, VALIDATOR_COUNTS, log):
    pVv, pViv, pG, j = log
    
    # Extract generator values
    idxs = [MODEL_ENUM[m] for m in pGa]
    G = np.array([pGa[m] for m in pGa])
    Ghat = pG[idxs]
    errorG = np.sum(np.abs(Ghat - G))

    return G, Ghat, errorG, pVv, pViv


def extract_values_logs(pGa, VALIDATOR_COUNTS, logs):
    errors = []
    pG_combi, pVv_combi, pViv_combi = [], [], []
   
    # Extract generator values
    for (pv1, pv2, pg, combo) in logs:
        G, Ghat, errorG, pVv, pViv = extract_values_log(pGa, VALIDATOR_COUNTS, (pv1, pv2, pg, combo))
        pG_combi.append(Ghat)
        pVv_combi.append(pVv)
        pViv_combi.append(pViv)
        errors.append(errorG)

    # Find the mean, max, min errors
    m_idx = np.argsort(errors)[len(errors)//2]
    mean_pG = logs[m_idx][2]
    min_pG = logs[np.argmin(errors)][2]
    max_pG = logs[np.argmax(errors)][2]

    # Extract validator values. Take the mean across all combinations
    mean_pVv = np.mean(pVv_combi, axis=0)
    mean_pViv = np.mean(pViv_combi, axis=0)

    return min_pG, max_pG, mean_pG, mean_pVv, mean_pViv

# =========================
#   Estimate for a given k
# =========================

def estimate_probs(k, GV, pVva, pViva, pGa, idV, idG, w = [1, 0, 0]):
    final_loss_best = np.inf
    pVv, pViv, pG = {}, {}, {}

    NUM_VALIDATORS = GV.shape[1]
    NUM_GENERATORS = GV.shape[0]

    if PG_START == 'mean':
        pG_ = np.mean(GV, axis=1)
    elif PG_START == 'uniform':
        pG_ = np.random.uniform(0, 1, NUM_GENERATORS)
    else:
        pG_ = np.ones(NUM_GENERATORS) * PG_START

    if PV_START == 'uniform':
        pVv_hat_ = np.random.uniform(0, 1, NUM_VALIDATORS)
    else:
        pVv_hat_ = np.ones(NUM_VALIDATORS) * PV_START

    if PVIV_START == 'uniform':
        pViv_hat_ = np.random.uniform(0, 1, NUM_VALIDATORS)
    else:
        pViv_hat_ = np.ones(NUM_VALIDATORS) * PVIV_START

    for run_count in range (NUM_RUNS):
        if run_count != 0:
            pG_hat = pG_ + np.random.uniform(-ERR_EPSILON_PG/2, ERR_EPSILON_PG/2, NUM_GENERATORS)
            pVv_hat = pVv_hat_ + np.random.uniform(-ERR_EPSILON_PV/2, ERR_EPSILON_PV/2, NUM_VALIDATORS)
            pViv_hat = pViv_hat_ + np.random.uniform(-ERR_EPSILON_PIV/2, ERR_EPSILON_PIV/2, NUM_VALIDATORS)
        else:
            pG_hat, pVv_hat, pViv_hat = pG_, pVv_hat_, pViv_hat_

        pG_hat = np.clip(pG_hat, 0, 1)
        pVv_hat = np.clip(pVv_hat, 0, 1)
        pViv_hat = np.clip(pViv_hat, 0, 1)  

        x = np.concatenate([pVv_hat, pViv_hat, pG_hat])

        res = minimize(total_loss, x, args=(GV, pVva, pViva, pGa, idV, idG, w), bounds=[(0, 1)] * len(x), method='L-BFGS-B')
        if(res.success == False):
            print(f"\033[91mRun {run_count} failed to converge\033[0m")

        final_loss, final_loss_best, pVv_t, pViv_t, pG_t = write_loss_to_csv(
            k, res, GV, pVva, pViva, pGa, idV, idG, w, NUM_VALIDATORS, final_loss_best
        )

        if final_loss_best > final_loss:
            final_loss_best = final_loss
            pVv = pVv_t
            pViv = pViv_t
            pG = pG_t
        
    return pVv, pViv, pG

# =========================
#   REGRESS
# =========================

def regress(GV, pGa, PVVA, PVIVA, gens, k1, VALIDATOR_COUNTS, w=[1, 0, 0]):
    idG = list(combinations(pGa.keys(), k1))
    pGs = []

    max_errors = []
    avg_errors = []

    logs = []

    numComb = 0
    for j in idG:
        stats = np.zeros_like(VALIDATOR_COUNTS[0])

        for i in j:
            stats += VALIDATOR_COUNTS[gens.index(i)]

        if k1 == 0:
            pVva, pViva = {}, {}
            pVv, pViv, pG = estimate_probs(k1, GV, {}, {}, pGa, (), j, w=w)
        else:
            pVva  = {k: (stats[i][3])/(stats[i][2] + stats[i][3]) for i, k in enumerate(MODELS)}
            pViva = {k: (stats[i][0])/(stats[i][0] + stats[i][1]) for i, k in enumerate(MODELS)}

            pVv, pViv, pG = estimate_probs(k1, GV, pVva, pViva, pGa, MODELS, j, w=w)

        log = (pVv, pViv, pG, j)
        logs.append(log)
        pGs.append(pG)

        excluded = [candidate for candidate in gens if candidate not in j]
        pG_temp = np.array([pGa[m] for m in excluded])
        pG_hat_temp = np.array([pG[MODEL_ENUM[m]] for m in excluded])
        error_temp = np.abs(pG_hat_temp - pG_temp)

        avg_errors.append(100*np.mean(error_temp))
        max_errors.append(100*max(error_temp) if len(pG_temp) > 0 else 0)

        numComb += 1
        print_values_k(k1, numComb, j, pGa, VALIDATOR_COUNTS, log, max_errors, avg_errors, PVVA, PVIVA)
        

    min_pG, max_pG, mean_pG, mean_pVv, mean_pViv = extract_values_logs(pGa, VALIDATOR_COUNTS, logs)
    write_latex_table(k1, mean_pVv, mean_pViv, min_pG, max_pG, mean_pG, PVVA, PVIVA, pGa)

    print(f"\n\nk = {k1} | {numComb} combinations")
    print(f"Min G Error: {100*np.mean(min_pG)}")
    return pGs, max_errors, avg_errors, logs


# =========================
#   PLOT DATA
# =========================

def plot_data_no_exclude(all_logs, pGa, pG_mean):
    means = []
    maxs = []
    base_mean = []
    all_errors = []
    base_max = []

    line_styles = [
        '-', '--', '-.', ':',
        (0, (5,2)), (0, (5,1)), (0, (3,1,1,1)), (0, (1,2)), (0, (3,5,1,5)), (0, (1,2)),
        (0, (2,3,2,3)), (0, (3,1,1,1,1,1)), (0, (4,2,1,2)), (0, (2,2,1,2,1,2)), (0, (3,2,1,2,1,2)), 
        (0, (4,2,1,2,1,2)), (0, (3,2,1,2,1,2,1,2)), (0, (4,2,1,2,1,2,1,2)), (0, (3,2,1,2,1,2,1,2,1,2)),
        (0, (4,2,1,2,1,2,1,2,1,2)), (0, (3,2,1,2,1,2,1,2,1,2,1,2)), (0, (4,2,1,2,1,2,1,2,1,2,1,2)), (0, (3,2,1,2,1,2,1,2,1,2,1,2,1,2)),
        (0, (4,2,1,2,1,2,1,2,1,2,1,2,1,2)), (0, (3,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2)), (0, (4,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2)),
        (0, (1, 1)), (0, (3, 3, 1, 3)), (0, (5, 5, 1, 5)), (0, (7, 7, 1, 7)),
        (0, (1, 1, 1, 1)), (0, (3, 3, 3, 3)), (0, (5, 5, 5, 5)), (0, (7, 7, 7, 7)),
        (0, (1, 1, 3, 1)), (0, (3, 3, 5, 3)), (0, (5, 5, 7, 5)), (0, (7, 7, 9, 7)),
        (0, (1, 1, 1, 3)), (0, (3, 3, 3, 5)), (0, (5, 5, 5, 7)), (0, (7, 7, 7, 9)),
        (0, (1, 1, 3, 3)), (0, (3, 3, 5, 5)), (0, (5, 5, 7, 7)), (0, (7, 7, 9, 9)),
        (0, (1, 1, 5, 1)), (0, (3, 3, 7, 3)), (0, (5, 5, 9, 5)), (0, (7, 7, 11, 7)),
        (0, (1, 1, 7, 1)), (0, (3, 3, 9, 3)), (0, (5, 5, 11, 5)), (0, (7, 7, 13, 7)),
        (0, (1, 1, 9, 1)), (0, (3, 3, 11, 3)), (0, (5, 5, 13, 5)), (0, (7, 7, 15, 7))
    ]
    cmap = plt.get_cmap('tab10', len(all_logs))

    fig, ax = plt.subplots()
    for k, log in enumerate(all_logs):
        # if 0 <= k < len(all_logs)-1:
        if k in [0, 1, 2]:
            tmp_errors = []
            for (v1, v2, pG_est, _) in log:
                idxs = [MODEL_ENUM[m] for m in pGa]
                target = [pGa[m] for m in pGa]
                tmp_errors.append(np.sum(np.abs(pG_est[idxs] - target)))
            m_idx = np.argsort(tmp_errors)[len(tmp_errors)//2]
            median_log = log[m_idx][2]
            ax.plot(median_log*100, label=f'Median k = {k}', linestyle=line_styles[k], color=cmap(k))

    ax.plot(pG_mean*100, color='black', label='Mean Prediction')
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels(MODEL_NAMES, rotation=50, fontsize=12)
    ax.tick_params(axis='x', length=10)

    # for i, key in enumerate(pGa.keys()):
    #     ax.scatter(MODEL_ENUM[key], pGa[key]*100, color='black')

    for i, key in enumerate(pGa.keys()):
        if i == 0:
            ax.scatter(MODEL_ENUM[key], pGa[key]*100, color='black', label='Ground Truth')
        else:
            ax.scatter(MODEL_ENUM[key], pGa[key]*100, color='black')

    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{PATH_REGRESSION}/all_regressor{VALIDATOR_REPAIR_SUFFIX}.pdf', format='pdf')


    fig, ax = plt.subplots()
    for k, log in enumerate(all_logs):
        # if 0 <= k < len(all_logs)-1:
        if k in [0, 2]:
            tmp_errors = []
            for (v1, v2, pG_est, _) in log:
                idxs = [MODEL_ENUM[m] for m in pGa]
                target = [pGa[m] for m in pGa]
                tmp_errors.append(np.sum(np.abs(pG_est[idxs] - target)))
            min_log = log[np.argmin(tmp_errors)][2]
            ax.plot(min_log*100, label=f'Peek: {k}', linestyle=line_styles[k], color=cmap(k))

    ax.plot(pG_mean*100, color='black', label='Mean')
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels(MODEL_NAMES, rotation=50, fontsize=12)
    ax.tick_params(axis='x', length=10)

    for i, key in enumerate(pGa.keys()):
        ax.scatter(MODEL_ENUM[key], pGa[key]*100, color='black')

    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{PATH_REGRESSION}/all_regressor_min{VALIDATOR_REPAIR_SUFFIX}.pdf', format='pdf')


    fig, ax = plt.subplots()
    for k, log in enumerate(all_logs):
        # if 0 <= k < len(all_logs)-1:
        if k in [0, 2]:
            tmp_errors = []
            for (v1, v2, pG_est, _) in log:
                idxs = [MODEL_ENUM[m] for m in pGa]
                target = [pGa[m] for m in pGa]
                tmp_errors.append(np.sum(np.abs(pG_est[idxs] - target)))
            min_log = log[np.argmax(tmp_errors)][2]
            ax.plot(min_log*100, label=f'Peek: {k}', linestyle=line_styles[k], color=cmap(k))

    ax.plot(pG_mean*100, color='black', label='Mean')
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels(MODEL_NAMES, rotation=50, fontsize=12)
    ax.tick_params(axis='x', length=10)

    for i, key in enumerate(pGa.keys()):
        ax.scatter(MODEL_ENUM[key], pGa[key]*100, color='black')

    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{PATH_REGRESSION}/all_regressor_max{VALIDATOR_REPAIR_SUFFIX}.pdf', format='pdf')


    for k, log in enumerate(all_logs):
        means.append([])
        maxs.append([])
        all_errors.append([])
        base_mean.append([])
        base_max.append([])

        fig, ax = plt.subplots()

        if 0 < k < len(all_logs)-1:
            ax.plot(np.mean([combi[2]*100 for combi in log], axis=0), color='red', label='Mean peek')

        for index, combi in enumerate(log):
            pVv, pViv, pG, j = combi

            if k == 0:
                ax.plot(pG * 100, linestyle=line_styles[index], color=cmap(index), label='k=0')
            else:
                ax.plot(pG * 100, linestyle=line_styles[index], color=cmap(index), label=str(j))

            excluded = [candidate for candidate in pGa.keys() if candidate not in j]

            avg_error, pG_mean_error = 0, 0
            for m in excluded:
                all_errors[k].append(abs(pG[MODEL_ENUM[m]] - pGa[m]))
                avg_error += abs(pG[MODEL_ENUM[m]] - pGa[m])
                pG_mean_error += abs(pG_mean[MODEL_ENUM[m]] - pGa[m])

            avg_error = avg_error/len(excluded) if len(excluded) > 0 else 0
            pG_mean_error = pG_mean_error/len(excluded) if len(excluded) > 0 else 0
            max_error = max(abs(pG[MODEL_ENUM[m]] - pGa[m]) for m in excluded) if len(excluded) > 0 else 0
            base_max_error = max(abs(pG_mean[MODEL_ENUM[m]] - pGa[m]) for m in excluded) if len(excluded) > 0 else 0

            if k == 0:
                for m in excluded:
                    print(abs(100* (pG_mean[MODEL_ENUM[m]] - pGa[m])))
                print(pG_mean_error/len(excluded) if len(excluded) > 0 else 0)

            
            means[k].append(avg_error)
            maxs[k].append(max_error)
            base_mean[k].append(pG_mean_error)
            base_max[k].append(base_max_error)
            
        ax.plot(pG_mean*100, color='black', label='Mean')

        ax.set_xticks(range(len(MODELS)))
        ax.set_xticklabels(MODEL_NAMES, rotation=50, fontsize=12)

        for i, key in enumerate(pGa.keys()):
            ax.scatter(MODEL_ENUM[key], pGa[key]*100, color='black')

        plt.tight_layout()
        plt.savefig(f'{PATH_REGRESSION}/regressor_{k}{VALIDATOR_REPAIR_SUFFIX}.pdf', format='pdf')


    fig, ax = plt.subplots()

    data = [[val * 100 for val in sublist] for sublist in maxs[:-1]]
    means_vals = [np.mean(d) for d in data]
    # stds = [np.std(d) for d in data]
    yerrs = np.transpose([(np.mean(d) - np.min(d), np.max(d) - np.mean(d)) for d in data])
    x_pos = range(1, len(all_logs))

    ax.errorbar(x_pos, means_vals, yerr=yerrs, fmt='-o', label='Regression', color='black', capsize=5)
    ax.set_xticks(x_pos)
    ax.fill_between(range(1, len(all_logs)), LLM_VALIDATOR_MAXERR_RANGE[0]*100, LLM_VALIDATOR_MAXERR_RANGE[1]*100, color='blue', alpha=0.1, label='Individual LLM')
    ax.axhline(y=ENSEMBLE_MAJORITY_MAXERR*100, color='red', label='Majority Ensemble', linestyle='-.')
    ax.axhline(y=ENSEMBLE_BEST_MAXERR*100, color='green', label='Best Ensemble', linestyle=':')
    ax.axhline(y=base_max[0][0]*100, color='gray', label='Mean Prediction', linestyle='--')

    ax.set_xticklabels(range(0, len(all_logs) - 1), fontsize=14)
    ax.set_xlabel('k', fontsize=16)
    ax.set_ylabel('Maximum Error', fontsize=16)
    ax.tick_params(axis='y', labelsize=14)
    plt.tight_layout()

    handles, labels = plt.gca().get_legend_handles_labels()
    order = [0,4,1,3,2]
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], fontsize=16, loc='upper right')

    plt.savefig(f'{PATH_REGRESSION}/regressor_max_comparison{VALIDATOR_REPAIR_SUFFIX}.pdf', format='pdf')

    
    fig, ax = plt.subplots()
    ax.axhline(y=base_max[0][0]*100, color='gray', label='Baseline Mean', linestyle='--')
    ax.boxplot(data, positions=x_pos, showmeans=True)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(range(0, len(all_logs) - 1))
    ax.set_xlabel('k')
    ax.set_ylabel('Maximum Error')
    plt.tight_layout()
    plt.legend()
    plt.savefig(f'{PATH_REGRESSION}/regressor_max_comparison_boxplot{VALIDATOR_REPAIR_SUFFIX}.pdf', format='pdf')


    print(base_max[0][0]*100, end=' ')
    # for m in means_vals:
    #     print(f"{m:.3f}", end='\\% & ')
    # print()

    print(base_mean[0][0]*100)

    data_means = [[val * 100 for val in sublist] for sublist in means[:-1]]
    means_vals_means = [np.mean(d) for d in data_means]
    yerrs_means = np.transpose([(np.mean(d) - np.min(d), np.max(d) - np.mean(d)) for d in data_means])

    for m, o1 in zip(means_vals_means, yerrs_means[0]):
        print(f"{m - o1:.1f}", end='\\% & ')
    print()

    for m in means_vals_means:
        print(f"{m:.1f}", end='\\% & ')
    print()

    for m, o1 in zip(means_vals_means, yerrs_means[1]):
        print(f"{m + o1:.1f}", end='\\% & ')
    print()
    print()

    for ma, o2 in zip(means_vals, yerrs[0]):
        print(f"{ma - o2:.1f}", end='\\% & ')
    print()

    for ma in means_vals:
        print(f"{ma:.1f}", end='\\% & ')
    print()

    for ma, o2 in zip(means_vals, yerrs[1]):
        print(f"{ma + o2:.1f}", end='\\% & ')
    print()
    print()

    fig, ax = plt.subplots()
    ax.errorbar(x_pos, means_vals_means, yerr=yerrs_means, fmt='-o', label='Regression', color='black', capsize=5)
    ax.axhline(y=ENSEMBLE_MAJORITY_MEANERR*100, color='red', label='Majority Ensemble', linestyle='-.')
    ax.axhline(y=ENSEMBLE_BEST_MEANERR*100, color='green', label='Best Ensemble', linestyle=':')
    ax.fill_between(range(1, len(all_logs)), LLM_VALIDATOR_MEANERR_RANGE[0]*100, LLM_VALIDATOR_MEANERR_RANGE[1]*100, color='blue', alpha=0.1, label='Individual LLM')
    ax.axhline(y=base_mean[0][0]*100, color='gray', label='Mean Prediction', linestyle='--')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(range(0, len(all_logs) - 1), fontsize=14)
    ax.set_xlabel('k', fontsize=16)
    ax.set_ylabel('Mean Error', fontsize=16)
    ax.tick_params(axis='y', labelsize=14)
    plt.tight_layout()
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [0,4,1,3,2]
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], fontsize=16, loc='upper right')
    plt.savefig(f'{PATH_REGRESSION}/regressor_mean_comparison{VALIDATOR_REPAIR_SUFFIX}.pdf', format='pdf')

# =========================
#   MAIN
# =========================
if __name__ == '__main__':
    GV, pGa, VALIDATOR_COUNTS, PVVA, PVIVA = read_data()

    redo = True
    if redo:
        all_max_errors = []
        all_avg_errors = []
        all_logs = []

        futures = {}
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            for k in K_LIST:
                futures[executor.submit(regress, GV, pGa, PVVA, PVIVA, GENS, k, VALIDATOR_COUNTS, w=WEIGHTS)] = k

        results = []
        for future in as_completed(futures):
            k = futures[future]
            results.append((k, future.result()))

        results.sort(key=lambda x: x[0])

        for k, (pGs, errors, avg_error, logs) in results:
            all_max_errors.append(errors)
            all_logs.append(logs)
            all_avg_errors.append(avg_error)
            print("Done with k =", k)


        with open(PATH_LOGS + 'all_logs.pkl', 'wb') as f:
            pickle.dump(all_logs, f)
    else:
        with open(PATH_LOGS + 'all_logs.pkl', 'rb') as f:
            all_logs = pickle.load(f)
    
    print('============================================')
    plot_data_no_exclude(all_logs, pGa, np.mean(GV, axis=1))
    print('============================================')

    pretty_print_into_file('regression_max_errors', all_max_errors, fpathRegressionSummary, comment='Max errors for all k values')
    pretty_print_into_file('regression_avg_errors', all_avg_errors, fpathRegressionSummary, comment='Mean errors for all k values')

    print('Baseline: ', np.round(100*np.mean(GV, axis=1), 1))