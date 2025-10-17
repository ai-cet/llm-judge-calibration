import config
from dataframe import generator, validator
from latex import predMatrixTex, validatorTex

def read_dfs():
    dfs = generator.get_gold_data(config.MODELS_GEN, config.pathGold, config.fnameGoldTable)
    df_gens = generator.get_gen_data(config.MODELS, config.pathGold)
    dfs = validator.get_validator_data(dfs, df_gens, config.MODELS, config.MODELS_GEN, config.pathValidator)
    df_val_tpr_tnr = validator.get_tpr_tnr_df(dfs, config.MODELS, config.MODELS_GEN)

    return dfs, df_gens, df_val_tpr_tnr

def write_latex(dfs, df_gens, df_val_tpr_tnr):
    genMeanHash, genPrecHash = predMatrixTex.generate_predicted_matrix_latex(dfs, config.MODELS, config.mapping_latex, config.fnamePredMatrix)
    predMatrixTex.generate_predicted_matrix_error(dfs, genMeanHash, genPrecHash, config.MODELS, config.MODELS_GEN, config.mapping_latex, config.fnamePredMatrix)
    validatorTex.generate_validator_latex(dfs, config.tpr_tnr_genModel, config.MODELS, config.mapping_latex, config.fnameTprTnrTex)

if __name__ == "__main__":
    dfs, df_gens, df_val_tpr_tnr = read_dfs()
    write_latex(dfs, df_gens, df_val_tpr_tnr)