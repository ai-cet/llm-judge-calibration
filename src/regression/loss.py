

from src.regression.config import GENS, GV_CONST, IS_WRITE_LOGS_LOSS, IS_WRITE_LOGS_ITER, MODEL_ENUM, MODELS, PATH_LOGS, PATH_LOGS_LOSS_ITER, PATH_LOGS_PREDICTED
from src.regression.config import LOSS_PRED, LOSS_REG

import numpy as np
import os
from enum import Enum

# =========================
#   PREDICTION LOSS
# =========================

def loss_pred_crossEntropy_weighted(pVv, pViv, pG, GV, alpha=0.3):
    """
    Calculate the cross-entropy loss between the predicted and actual gene-variant matrices.
    
    This function computes a binary cross-entropy loss between the generator precision 
    prediction by validator (GV) and its estimate by regression method (GV_hat). 
    The GV_hat matrix is constructed using outer products of generator and validator performance vectors.
    
    Parameters:
    -----------
    pVv : numpy.ndarray
        Probability vector of a validator classifying output as valid.
    pViv : numpy.ndarray
        Probability vector of a validator classifying output as invalid.
    pG : numpy.ndarray
        Probability vector for generator output being valid.
    GV : numpy.ndarray
        Generator precision prediction by validator matrix.

    alpha : float, optional
        Weighting factor for the cross-entropy loss. Default is 0.3.
    
    Returns:
    --------
    float
        Negative mean of the cross-entropy values, representing the loss.
        
    Notes:
    ------
    A small epsilon (1e-9) is added to avoid log(0) errors through clipping.
    """
    GV_hat = np.outer(pG, pVv) + np.outer((1 - pG), (1 - pViv))
    epsilon = 1e-9
    return -np.mean([
        alpha*GV[i, j]*np.log(np.clip(GV_hat[i, j], epsilon, 1))
        + (1-alpha)*(1 - GV[i, j])*np.log(np.clip(1 - GV_hat[i, j], epsilon, 1))
        for i in range(GV.shape[0])
        for j in range(GV.shape[1])
    ])

def loss_pred_focal(pVv, pViv, pG, GV, gamma=2.0):
    """
    Calculate the focal loss between the predicted and actual gene-variant matrices.
    
    This function computes a focal loss, which is a modified version of cross-entropy loss 
    that focuses more on hard-to-classify examples. The GV_hat matrix is constructed using 
    outer products of generator and validator performance vectors.
    
    Parameters:
    -----------
    pVv : numpy.ndarray
        Probability vector of a validator classifying output as valid.
    pViv : numpy.ndarray
        Probability vector of a validator classifying output as invalid.
    pG : numpy.ndarray
        Probability vector for generator output being valid.
    GV : numpy.ndarray
        Generator precision prediction by validator matrix.
    
    gamma : float, optional
        Focusing parameter for the focal loss. Default is 2.0.
    
    Returns:
    --------
    float
        Negative mean of the focal loss values, representing the loss.
        
    Notes:
    ------
    A small epsilon (1e-9) is added to avoid log(0) errors through clipping.
    """
    GV_hat = np.outer(pG, pVv) + np.outer((1 - pG), (1 - pViv))
    epsilon = 1e-9
    return -50*np.mean([
        (GV[i, j] * (1 - GV_hat[i, j])**gamma * np.log(np.clip(GV_hat[i, j], epsilon, 1)) +
         (1 - GV[i, j]) * GV_hat[i, j]**gamma * np.log(np.clip(1 - GV_hat[i, j], epsilon, 1)))
        for i in range(GV.shape[0])
        for j in range(GV.shape[1])
    ])

def loss_pred_crossEntropy(pVv, pViv, pG, GV):
    """
    Calculate the cross-entropy loss between the predicted and actual gene-variant matrices.
    
    This function computes a binary cross-entropy loss between the generator precision 
    prediction by validator (GV) and its estimate by regression method (GV_hat). 
    The GV_hat matrix is constructed using outer products of generator and validator performance vectors.
    
    Parameters:
    -----------
    pVv : numpy.ndarray
        Probability vector of a validator classifying output as valid.
    pViv : numpy.ndarray
        Probability vector of a validator classifying output as invalid.
    pG : numpy.ndarray
        Probability vector for generator output being valid.
    GV : numpy.ndarray
        Generator precision prediction by validator matrix.
    
    Returns:
    --------
    float
        Negative mean of the cross-entropy values, representing the loss.
        
    Notes:
    ------
    A small epsilon (1e-9) is added to avoid log(0) errors through clipping.
    """
    GV_hat = np.outer(pG, pVv) + np.outer((1 - pG), (1 - pViv))
    epsilon = 1e-9
    return -np.mean([
        GV[i, j]*np.log(np.clip(GV_hat[i, j], epsilon, 1))
        + (1 - GV[i, j])*np.log(np.clip(1 - GV_hat[i, j], epsilon, 1))
        for i in range(GV.shape[0])
        for j in range(GV.shape[1])
    ])

def loss_pred_mse(pVv, pViv, pG, GV):
    """
    Calculate the mean squared error between the predicted and actual gene-variant matrices.

    This function computes the mean squared error between the generator precision
    prediction by validator (GV) and its estimate by regression method (GV_hat).
    The GV_hat matrix is constructed using outer products of generator and validator performance vectors.
    
    Parameters:
    -----------
    pVv : numpy.ndarray
        Probability vector of a validator classifying output as valid.
    pViv : numpy.ndarray
        Probability vector of a validator classifying output as invalid.
    pG : numpy.ndarray
        Probability vector for generator output being valid.
    GV : numpy.ndarray
        Generator precision prediction by validator matrix.
    
    Returns:
    --------
    float
        Mean squared error between GV and GV_hat.
    """
    GV_hat = np.outer(pG, pVv) + np.outer((1 - pG), (1 - pViv))
    return np.mean((GV_hat - GV)**2)

def loss_pred_mae(pVv, pViv, pG, GV):
    """
    Calculate the mean absolute error between the predicted and actual gene-variant matrices.
    
    This function computes the mean absolute error between the generator precision 
    prediction by validator (GV) and its estimate by regression method (GV_hat). 
    The GV_hat matrix is constructed using outer products of generator and validator performance vectors.
    
    Parameters:
    -----------
    pVv : numpy.ndarray
        Probability vector of a validator classifying output as valid.
    pViv : numpy.ndarray
        Probability vector of a validator classifying output as invalid.
    pG : numpy.ndarray
        Probability vector for generator output being valid.
    GV : numpy.ndarray
        Generator precision prediction by validator matrix.
    
    Returns:
    --------
    float
        Mean absolute error between GV and GV_hat.
    """
    GV_hat = np.outer(pG, pVv) + np.outer((1 - pG), (1 - pViv))
    return np.mean(np.abs(GV_hat - GV))
    
def loss_pred_huber(pVv, pViv, pG, GV):
    """
    Calculate the Huber loss between the predicted and actual gene-variant matrices.
    
    This function computes the Huber loss between the generator precision 
    prediction by validator (GV) and its estimate by regression method (GV_hat). 
    The GV_hat matrix is constructed using outer products of generator and validator performance vectors.
    
    Parameters:
    -----------
    pVv : numpy.ndarray
        Probability vector of a validator classifying output as valid.
    pViv : numpy.ndarray
        Probability vector of a validator classifying output as invalid.
    pG : numpy.ndarray
        Probability vector for generator output being valid.
    GV : numpy.ndarray
        Generator precision prediction by validator matrix.
    
    Returns:
    --------
    float
        Huber loss between GV and GV_hat.
    """
    GV_hat = np.outer(pG, pVv) + np.outer((1 - pG), (1 - pViv))
    delta = 0.01
    return np.mean(np.where(np.abs(GV_hat - GV) < delta,
                             0.5 * (GV_hat - GV)**2,
                             delta * (np.abs(GV_hat - GV) - 0.5 * delta)))

def loss_pred_rmse(pVv, pViv, pG, GV):
    """
    Calculate the root mean squared error (RMSE) between the predicted and actual gene-variant matrices.
    
    This function computes the RMSE between the generator precision prediction by validator (GV) 
    and its estimate by regression method (GV_hat). The GV_hat matrix is constructed using outer 
    products of generator and validator performance vectors.
    
    Parameters:
    -----------
    pVv : numpy.ndarray
        Probability vector of a validator classifying output as valid.
    pViv : numpy.ndarray
        Probability vector of a validator classifying output as invalid.
    pG : numpy.ndarray
        Probability vector for generator output being valid.
    GV : numpy.ndarray
        Generator precision prediction by validator matrix.
    
    Returns:
    --------
    float
        RMSE between GV and GV_hat.
    """
    GV_hat = np.outer(pG, pVv) + np.outer((1 - pG), (1 - pViv))
    return np.sqrt(np.mean((GV_hat - GV)**2))


def loss_pred(pVv, pViv, pG, GV):
    if LOSS_PRED == 'crossEntropy':
        return loss_pred_crossEntropy(pVv, pViv, pG, GV)
    elif LOSS_PRED == 'mse':
        return loss_pred_mse(pVv, pViv, pG, GV)
    elif LOSS_PRED == 'rmse':
        return loss_pred_rmse(pVv, pViv, pG, GV)
    elif LOSS_PRED == 'mae':
        return loss_pred_mae(pVv, pViv, pG, GV)
    elif LOSS_PRED == 'huber':
        return loss_pred_huber(pVv, pViv, pG, GV)
    elif LOSS_PRED == 'crossEntropy_weighted':
        return loss_pred_crossEntropy_weighted(pVv, pViv, pG, GV)
    elif LOSS_PRED == 'focal':
        return loss_pred_focal(pVv, pViv, pG, GV)
    elif LOSS_PRED == 'custom':
        return 0.5*loss_pred_crossEntropy(pVv, pViv, pG, GV) + loss_pred_rmse(pVv, pViv, pG, GV)
    else:
        raise ValueError(f"Unknown loss function: {LOSS_PRED}")

# =========================
#   Regularization LOSS
# =========================

def loss_reg_rmse(p_hat, p, id):
    """
    Calculate the root mean squared error (RMSE) between predicted and actual values.

    This function computes the RMSE for a specific subset of indices, comparing
    predictions from a model to the actual values.

    Parameters:
    ----------
    p_hat : dict
        Dictionary mapping model names to their predictions.
    p : list or array-like
        The actual/ground truth values.
    id : list or array-like
        Indices for which to calculate the error. If empty, returns 0.

    Returns:
    -------
    float
        The RMSE between predictions and actual values for the specified indices.
        Returns 0 if `id` is empty.
    """
    return np.sqrt(np.mean([(p[i] - p_hat[MODEL_ENUM[i]])**2 for i in id]) if len(id) > 0 else 0)

def loss_reg_mae(p_hat, p, id):
    """
    Calculate the mean absolute error (MAE) between predicted and actual values.

    This function computes the MAE for a specific subset of indices, comparing
    predictions from a model to the actual values.

    Parameters:
    ----------
    p_hat : dict
        Dictionary mapping model names to their predictions.
    p : list or array-like
        The actual/ground truth values.
    id : list or array-like
        Indices for which to calculate the error. If empty, returns 0.

    Returns:
    -------
    float
        The MAE between predictions and actual values for the specified indices.
        Returns 0 if `id` is empty.
    """
    return np.mean([abs(p[i] - p_hat[MODEL_ENUM[i]]) for i in id]) if len(id) > 0 else 0

def loss_reg_huber(p_hat, p, id):
    """
    Calculate the Huber loss between predicted and actual values.

    This function computes the Huber loss for a specific subset of indices, comparing
    predictions from a model to the actual values.

    Parameters:
    ----------
    p_hat : dict
        Dictionary mapping model names to their predictions.
    p : list or array-like
        The actual/ground truth values.
    id : list or array-like
        Indices for which to calculate the error. If empty, returns 0.

    Returns:
    -------
    float
        The Huber loss between predictions and actual values for the specified indices.
        Returns 0 if `id` is empty.
    """
    delta = 0.05
    return np.mean([np.where(abs(p[i] - p_hat[MODEL_ENUM[i]]) < delta,
                             0.5 * (p[i] - p_hat[MODEL_ENUM[i]])**2,
                             delta * (abs(p[i] - p_hat[MODEL_ENUM[i]]) - 0.5 * delta)) for i in id]) if len(id) > 0 else 0

def loss_reg(p_hat, p, id):
    """
    Calculate the loss between predicted and actual values.

    This function computes the loss for a specific subset of indices, comparing
    predictions from a model to the actual values.

    Parameters:
    ----------
    p_hat : dict
        Dictionary mapping model names to their predictions.
    p : list or array-like
        The actual/ground truth values.
    id : list or array-like
        Indices for which to calculate the error. If empty, returns 0.

    Returns:
    -------
    float
        The loss between predictions and actual values for the specified indices.
        Returns 0 if `id` is empty.
    """
    if LOSS_REG == 'rmse':
        return loss_reg_rmse(p_hat, p, id)
    elif LOSS_REG == 'mae':
        return loss_reg_mae(p_hat, p, id)
    elif LOSS_REG == 'huber':
        return loss_reg_huber(p_hat, p, id)
    else:
        raise ValueError(f"Unknown loss function: {LOSS_REG}")


# =========================
#   TOTAL LOSS
# =========================

def total_loss(x, GV, pVva, pViva, pGa, idV, idG, w):
    NUM_VALIDATORS = GV.shape[1]
    NUM_GENERATORS = GV.shape[0]
    l1 = loss_pred(x[:NUM_VALIDATORS], x[NUM_VALIDATORS:2*NUM_VALIDATORS], x[2*NUM_VALIDATORS:], GV)
    
    l2 = loss_reg(x[2*NUM_VALIDATORS:], pGa, idG)
    l3_1 = loss_reg(x[:NUM_VALIDATORS], pVva, idV)
    l3_2 = loss_reg(x[NUM_VALIDATORS:2*NUM_VALIDATORS], pViva, idV)

    if IS_WRITE_LOGS_ITER:
        write_iterLoss_csv(x, pGa, pVva, pViva, idG, idV)

    return l1 + w[0]*l2 + w[1]*l3_1 + w[2]*l3_2

# =========================
#   Write to CSV
# =========================

GLOBAL_LOSS_COUNT = 0

# Initialize CSV file for iteration losses
if not os.path.exists(PATH_LOGS_LOSS_ITER):
    os.makedirs(PATH_LOGS_LOSS_ITER)

path_iter_loss_csv = f"{PATH_LOGS_LOSS_ITER}iter_losses.csv"
if IS_WRITE_LOGS_ITER:
    with open(path_iter_loss_csv, 'w') as f:
        f.write("iter,train_loss_GV,train_loss_g,train_loss_vv,train_loss_viv,test_loss_GV,test_loss_g,test_loss_vv,test_loss_viv\n")

def write_iterLoss_csv(x, pGa, pVva, pViva, idG, idV):
    """
    Write the iteration loss to a CSV file.

    Parameters:
    ----------
    x : numpy.ndarray
        The current parameter values.
    pGa : dict
        Dictionary mapping model names to their predictions.
    pVva : dict
        Dictionary mapping model names to their predictions.
    idG : list
        Indices for generators.
    idV : list
        Indices for validators.
    """
    global GLOBAL_LOSS_COUNT
    GLOBAL_LOSS_COUNT += 1

    NUM_VALIDATORS = GV_CONST.shape[1]

    # find the hats
    pVv_hat = x[:NUM_VALIDATORS]
    pViv_hat = x[NUM_VALIDATORS:2*NUM_VALIDATORS]
    pG_hat = x[2*NUM_VALIDATORS:]

    # Test set
    idG_test = GENS
    idVV_test = MODELS
    idVIV_test = MODELS

    # Calculate training loss
    loss_train_GV = loss_pred(pVv_hat, pViv_hat, pG_hat, GV_CONST)
    loss_train_g = loss_reg(pG_hat, pGa, idG)
    loss_train_vv = loss_reg(pVv_hat, pVva, idV)
    loss_train_viv = loss_reg(pViv_hat, pViva, idV)

    # Calculate test loss
    with open(f'{PATH_LOGS}temp.txt', 'w') as f:
        f.write(f"idVV_test: {idVV_test}\n")
        f.write(f"pVva: {pVva}\n")
        f.write(f"pVv_hat: {pVv_hat}\n")

    loss_test_GV = loss_pred(pVv_hat, pViv_hat, pG_hat, GV_CONST)
    loss_test_g = loss_reg(pG_hat, pGa, idG_test)
    loss_test_vv = loss_reg(pVv_hat, pVva, idVV_test)
    loss_test_viv = loss_reg(pViv_hat, pViva, idVIV_test)

    # Append the current results
    with open(path_iter_loss_csv, 'a') as f:
        f.write(f"{GLOBAL_LOSS_COUNT},{loss_train_GV},{loss_train_g},{loss_train_vv},{loss_train_viv},{loss_test_GV},{loss_test_g},{loss_test_vv},{loss_test_viv}\n")

    # Write detailed results to a file with actual vs predicted values
    detail_path = f"{PATH_LOGS_PREDICTED}iter_details.csv"
    with open(detail_path, 'w') as f:
        # Write header
        f.write("type,model_idx,actual,predicted,loss\n")
        f.write(str(list(pGa.values())) + "\n")
        f.write(str(list(pG_hat)) + "\n")
        
        # Write Generator predictions
        for i in range(len(pG_hat)):
            if i in idG:  # Training set
                actual = pGa[i]
                pred = pG_hat[i]
                loss = (pred - actual)**2
                f.write(f"generator,{i},{actual},{pred},{loss}\n")
        
        # Write Validator (valid) predictions
        for i in range(len(pVv_hat)):
            if i in idV:  # Training set
                actual = pVva[i]
                pred = pVv_hat[i]
                loss = (pred - actual)**2
                f.write(f"validator_valid,{i},{actual},{pred},{loss}\n")
        
        # Write Validator (invalid) predictions
        for i in range(len(pViv_hat)):
            if i in idV:  # Training set
                actual = pViva[i]
                pred = pViv_hat[i]
                loss = (pred - actual)**2
                f.write(f"validator_invalid,{i},{actual},{pred},{loss}\n")
        
        # Write GV matrix predictions to a separate text file
        matrix_path = f"{PATH_LOGS_PREDICTED}iter_matrices.txt"
        with open(matrix_path, 'w') as mf:
            # Construct predicted GV matrix
            G_hat = np.outer(pG_hat, pVv_hat) + np.outer((1 - pG_hat), (1 - pViv_hat))
            
            # Write actual GV matrix
            mf.write("Actual GV Matrix:\n")
            np.savetxt(mf, GV_CONST, fmt='%.4f', delimiter=' ')
            mf.write("\n\n")
            
            # Write predicted GV matrix
            mf.write("Predicted GV Matrix:\n")
            np.savetxt(mf, G_hat, fmt='%.4f', delimiter=' ')
            mf.write("\n\n")
            
            # Write loss matrix (squared error)
            mf.write("Loss Matrix (MSE):\n")
            loss_matrix = (G_hat - GV_CONST)**2
            np.savetxt(mf, loss_matrix, fmt='%.4f', delimiter=' ')

    # print(f"{GLOBAL_LOSS_COUNT}. Loss GV: {l1}, pG: {l2}, pV+: {l3_1}, pV-: {l3_2}")


# Initialize CSV file for regression losses
path_overall_loss_csv = f"{PATH_LOGS_LOSS_ITER}regression_losses.csv"
if IS_WRITE_LOGS_LOSS:
    with open(path_overall_loss_csv, 'w') as f:
        f.write("k,models,final_loss,gv_loss,pg_loss,pvv_loss,pviv_loss,pg_mae,pvv_mae,pviv_mae,pg_err\n")

# Function to write the final loss to CSV
def write_loss_to_csv(k, res, GV, pVva, pViva, pGa, idV, idG, w, NUM_VALIDATORS, final_loss_best):
    # find the hats
    pVv_hat = res.x[:NUM_VALIDATORS]
    pViv_hat = res.x[NUM_VALIDATORS:2*NUM_VALIDATORS]
    pG_hat = res.x[2*NUM_VALIDATORS:]
    
    # Record the final loss values
    final_loss = total_loss(res.x, GV, pVva, pViva, pGa, idV, idG, w)
    gv_loss = loss_pred(pVv_hat, pViv_hat, pG_hat, GV)
    pg_loss = loss_reg(pG_hat, pGa, idG)
    pvv_loss = loss_reg(pVv_hat, pVva, idV)
    pviv_loss = loss_reg(pViv_hat, pViva, idV)


    pg_err = np.abs(np.array([pGa[model] for model in GENS]) - np.array([pG_hat[MODEL_ENUM[model]] for model in GENS]))
    pg_max = np.max(pg_err)
    pvv_max = np.max(np.abs(np.array([pVva[model] for model in MODELS]) - pVv_hat)) if len(pVva) > 0 else 0
    pviv_max = np.max(np.abs(np.array([pViva[model] for model in MODELS]) - pViv_hat)) if len(pViva) > 0 else 0
    # Convert the validation model combo to string
    models_str = ';'.join([str(m) for m in idG])

    # Append the current results
    if IS_WRITE_LOGS_LOSS:
        with open(path_overall_loss_csv, 'a') as f:
            f.write(f"{k},{models_str},{final_loss},{gv_loss},{pg_loss},{pvv_loss},{pviv_loss},{pg_max},{pvv_max},{pviv_max}, {pg_err}\n")

    
    return final_loss, final_loss_best, pVv_hat, pViv_hat, pG_hat