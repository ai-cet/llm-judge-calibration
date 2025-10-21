from src.config import VALIDATOR_REPAIR_NAME
import src.plots.elo_plots as elo_plots
import src.plots.ensemble_plots as ensemble_plots
import src.plots.validator_plots as validator_plots
import src.plots.validator_failure_plots as validator_failure_plots
import src.plots.repair_plots as repair_plots
import src.plots.slide_plots as slide_plots
import src.plots.regression_plots as regression_plots
import src.plots.regression_variance_plots as regression_variance_plots
import sys

# =========================
#   CUSTOM IMPORTS
# =========================
if VALIDATOR_REPAIR_NAME == 'fcnhp':
    from src.validation.generated_scripts.summary_fcnhp import (ensemble_max_errors, ensemble_tpr, ensemble_tnr, error_message_counts_validator)
    from src.validation.generated_scripts.llm_as_judge_fcnhp import validator_tpr, validator_tnr, GV
elif VALIDATOR_REPAIR_NAME == '':
    from src.validation.generated_scripts.summary_noRepair import (ensemble_max_errors, ensemble_tpr, ensemble_tnr, error_message_counts_validator)
    from src.validation.generated_scripts.llm_as_judge_noRepair import validator_tpr, validator_tnr, GV
else:
    raise ValueError(f"Unknown VALIDATOR_REPAIR_STR: {VALIDATOR_REPAIR_NAME}. Please check the configuration.")

# =========================
#   MAIN
# =========================
if __name__ == "__main__":
    regression_plots.regression_plot()

    # Elo vs Precision plot
    elo_plots.elo_vs_precision()
    elo_plots.elo_vs_tnr()

    # Plots for AI4X slides
    slide_plots.llm_release_plot()
    slide_plots.llm_release_plot_trendline()
    slide_plots.llm_release_plot_area()

    # Validator TPR vs TNR plot
    validator_plots.tpr_tnr_validator(validator_tpr, validator_tnr, ensemble_tpr, ensemble_tnr)
    validator_plots.gv_plot(GV)
    validator_plots.gv_boxplot(GV)

    # Validator failure plots
    validator_failure_plots.plot_failure_counts(error_message_counts_validator)
    validator_failure_plots.plot_validator_valid()
    validator_failure_plots.plot_validator_invalid()
    validator_failure_plots.plot_validator_valid_invalid_cumsum()

    # Ensemble plots
    ensemble_plots.tpr_tnr_ensemble(ensemble_tpr, ensemble_tnr, validator_tpr, validator_tnr)
    ensemble_plots.valid_invalid_error(ensemble_max_errors)

    # Repair plots
    repair_plots.repair_vs_error()

    # Variance plots
    regression_variance_plots.regression_variance()