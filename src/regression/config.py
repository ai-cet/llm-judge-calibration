import numpy as np

from src.config import pGa_CONST, MODELS_GEN, MODELS_VAL, MODELS_SHORT
from src.config import pathData, pathOutput, pathLogs, VALIDATOR_REPAIR_NAME, VALIDATOR_REPAIR_SUFFIX

# =========================
#   CUSTOM IMPORTS
# =========================
if VALIDATOR_REPAIR_NAME == 'fcnhp':
    from src.validation.generated_scripts.llm_as_judge_fcnhp import confusion_matrix_validators, validator_tpr, validator_tnr, GV
    from src.validation.generated_scripts.llm_as_judge_fcnhp import llm_judge_max_error_range, llm_judge_mean_error_range
    from src.validation.generated_scripts.llm_as_judge_fcnhp import ensemble_majority_max_error, ensemble_majority_mean_error
    from src.validation.generated_scripts.llm_as_judge_fcnhp import ensemble_best_max_error, ensemble_best_mean_error
elif VALIDATOR_REPAIR_NAME == 'fc':
    from src.validation.generated_scripts.llm_as_judge_fc import confusion_matrix_validators, validator_tpr, validator_tnr, GV
    from src.validation.generated_scripts.llm_as_judge_fc import llm_judge_max_error_range, llm_judge_mean_error_range
    from src.validation.generated_scripts.llm_as_judge_fc import ensemble_majority_max_error, ensemble_majority_mean_error
    from src.validation.generated_scripts.llm_as_judge_fc import ensemble_best_max_error, ensemble_best_mean_error
elif VALIDATOR_REPAIR_NAME == 'fcnh':
    from src.validation.generated_scripts.llm_as_judge_fcnh import confusion_matrix_validators, validator_tpr, validator_tnr, GV
    from src.validation.generated_scripts.llm_as_judge_fcnh import llm_judge_max_error_range, llm_judge_mean_error_range
    from src.validation.generated_scripts.llm_as_judge_fcnh import ensemble_majority_max_error, ensemble_majority_mean_error
    from src.validation.generated_scripts.llm_as_judge_fcnh import ensemble_best_max_error, ensemble_best_mean_error
elif VALIDATOR_REPAIR_NAME == '':
    from src.validation.generated_scripts.llm_as_judge_noRepair import confusion_matrix_validators, validator_tpr, validator_tnr, GV
    from src.validation.generated_scripts.llm_as_judge_noRepair import llm_judge_max_error_range, llm_judge_mean_error_range
    from src.validation.generated_scripts.llm_as_judge_noRepair import ensemble_majority_max_error, ensemble_majority_mean_error
    from src.validation.generated_scripts.llm_as_judge_noRepair import ensemble_best_max_error, ensemble_best_mean_error
else:
    raise ValueError(f"Unknown VALIDATOR_REPAIR_STR: {VALIDATOR_REPAIR_NAME}. Please check the configuration.")

# =========================
#   INIT
# =========================
COMPUTE_REFERENCE_VALUES = False # Set to True to compute reference values. False to use pre-computed hardcoded values.

# =========================
#   FILE PATHS
# =========================
PATH_DATA = pathData

PATH_OUTPUT = pathOutput
PATH_INTERIM = PATH_OUTPUT + 'interim/'
PATH_IMAGES = PATH_OUTPUT + 'images/'
PATH_REGRESSION = PATH_IMAGES + f'regression/{VALIDATOR_REPAIR_NAME}/'
PATH_LATEX = PATH_OUTPUT + 'latex/'
PATH_LOGS = pathLogs
PATH_LOGS_LOSS_ITER = PATH_LOGS + 'loss_iteration/'
PATH_LOGS_PREDICTED = PATH_LOGS + 'predicted/'

pathRegressionGenScripts = f'./src/regression/generated_scripts/'
fpathRegressionSummary = f'{pathRegressionGenScripts}/summary{VALIDATOR_REPAIR_SUFFIX}.py'

# =========================
#   MODEL CONSTANTS
# =========================

GENS = MODELS_GEN
MODELS = MODELS_VAL
MODEL_NAMES = list(MODELS_SHORT.values())

MODEL_ENUM = {model: i for i, model in enumerate(MODELS)}

# =========================
#   GENERATOR CONSTANTS
# =========================
pGa_CONST = pGa_CONST


# =========================
#   VALIDATOR CONSTANTS
# =========================
VALIDATOR_COUNTS_CONST = confusion_matrix_validators

# =========================
#   PVVA & PVIVA CONSTANTS
# =========================

PVVA_CONST = np.array(validator_tpr)
PVIVA_CONST = np.array(validator_tnr)

# =========================
#   GV MATRIX
# =========================
GV_CONST = np.array(GV)

# =========================
#   EXPERIMENT CONSTANTS
# =========================
NUM_RUNS = 10 # number of runs to min over
MAX_WORKERS = 32 # number of workers for parallelization
K_LIST = range(0, len(GENS)+1) # number of generators to peek at
# K_LIST = [5] 

# =========================
#   LOSS CONSTANTS
# =========================
# WEIGHTS = [0.1, 0.01, 0.1] # weights for each loss function
WEIGHTS = [2, 1, 10] # weights for each loss function
# WEIGHTS = [1, 1, 1] # weights for each loss function
LOSS_PRED = 'crossEntropy' # options are 'crossEntropy', 'rmse', 'mse', 'mae', 'huber', 'crossEntropy_weighted', 'focal'
LOSS_REG = 'rmse' # options are 'rmse', 'mae', 'huber'

# =========================
#   STARTING PARAMETERS
# =========================
PV_START = 0.5 # options are 'uniform', float
PVIV_START = 0.5 # options are 'uniform', float
PG_START = 'mean' # options are 'mean', 'uniform', float

# =========================
#   ERROR CONSTANTS
# =========================
ERR_EPSILON_PV = 0.01 # For adding to start point
ERR_EPSILON_PIV = 0.01 # For adding to start point
ERR_EPSILON_PG = 0.001 # For adding to start point

# =========================
#   LOGS & PRINTING
# =========================

COLOR_GREEN_DELTA = 0.02    # 2% tolerance
COLOR_YELLOW_DELTA = 0.05   # 5%
IS_WRITE_LOGS_LOSS = True
IS_WRITE_LOGS_ITER = False

# =========================
#   LLM JUDGE
# =========================
LLM_VALIDATOR_MAXERR_RANGE = llm_judge_max_error_range
LLM_VALIDATOR_MEANERR_RANGE = llm_judge_mean_error_range
ENSEMBLE_MAJORITY_MAXERR = ensemble_majority_max_error
ENSEMBLE_MAJORITY_MEANERR = ensemble_majority_mean_error
ENSEMBLE_BEST_MAXERR = ensemble_best_max_error
ENSEMBLE_BEST_MEANERR = ensemble_best_mean_error

# LLM_VALIDATOR_MAXERR_RANGE = (2.2704516880093117, 18.00125925925926)
# LLM_VALIDATOR_MEANERR_RANGE = (1.2914530870128726, 11.516197498754769)
# ENSEMBLE_MAXERR = 12.11
# ENSEMBLE_MEANERR = 4.51
