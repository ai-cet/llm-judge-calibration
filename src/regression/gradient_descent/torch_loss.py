import torch
import torch.nn as nn
import numpy as np
from src.regression.config import LOSS_PRED, LOSS_REG, MODEL_ENUM, MODELS, GENS


def loss_pred_crossEntropy_torch(pVv, pViv, pG, GV, alpha=0.3):
    """
    PyTorch version of cross-entropy loss between predicted and actual gene-variant matrices.
    """
    GV_hat = torch.outer(pG, pVv) + torch.outer((1 - pG), (1 - pViv))
    epsilon = 1e-9
    GV_hat_clipped = torch.clamp(GV_hat, epsilon, 1 - epsilon)
    
    loss = -(alpha * GV * torch.log(GV_hat_clipped) + 
             (1 - alpha) * (1 - GV) * torch.log(1 - GV_hat_clipped))
    return torch.mean(loss)


def loss_pred_focal_torch(pVv, pViv, pG, GV, gamma=2.0):
    """
    PyTorch version of focal loss between predicted and actual gene-variant matrices.
    """
    GV_hat = torch.outer(pG, pVv) + torch.outer((1 - pG), (1 - pViv))
    epsilon = 1e-9
    GV_hat_clipped = torch.clamp(GV_hat, epsilon, 1 - epsilon)
    
    loss = -(GV * (1 - GV_hat_clipped)**gamma * torch.log(GV_hat_clipped) +
             (1 - GV) * GV_hat_clipped**gamma * torch.log(1 - GV_hat_clipped))
    return 50 * torch.mean(loss)


def loss_pred_crossEntropy_weighted_torch(pVv, pViv, pG, GV, alpha=0.3):
    """
    PyTorch version of weighted cross-entropy loss.
    """
    return loss_pred_crossEntropy_torch(pVv, pViv, pG, GV, alpha)


def loss_pred_mse_torch(pVv, pViv, pG, GV):
    """
    PyTorch version of MSE loss between predicted and actual gene-variant matrices.
    """
    GV_hat = torch.outer(pG, pVv) + torch.outer((1 - pG), (1 - pViv))
    return torch.mean((GV_hat - GV)**2)


def loss_pred_mae_torch(pVv, pViv, pG, GV):
    """
    PyTorch version of MAE loss between predicted and actual gene-variant matrices.
    """
    GV_hat = torch.outer(pG, pVv) + torch.outer((1 - pG), (1 - pViv))
    return torch.mean(torch.abs(GV_hat - GV))


def loss_pred_rmse_torch(pVv, pViv, pG, GV):
    """
    PyTorch version of RMSE loss between predicted and actual gene-variant matrices.
    """
    GV_hat = torch.outer(pG, pVv) + torch.outer((1 - pG), (1 - pViv))
    return torch.sqrt(torch.mean((GV_hat - GV)**2))


def loss_pred_huber_torch(pVv, pViv, pG, GV):
    """
    PyTorch version of Huber loss between predicted and actual gene-variant matrices.
    """
    GV_hat = torch.outer(pG, pVv) + torch.outer((1 - pG), (1 - pViv))
    delta = 0.01
    diff = GV_hat - GV
    return torch.mean(torch.where(torch.abs(diff) < delta,
                                  0.5 * diff**2,
                                  delta * (torch.abs(diff) - 0.5 * delta)))


def loss_pred_torch(pVv, pViv, pG, GV):
    """
    PyTorch version of prediction loss dispatcher.
    """
    if LOSS_PRED == 'crossEntropy':
        return loss_pred_crossEntropy_torch(pVv, pViv, pG, GV)
    elif LOSS_PRED == 'mse':
        return loss_pred_mse_torch(pVv, pViv, pG, GV)
    elif LOSS_PRED == 'rmse':
        return loss_pred_rmse_torch(pVv, pViv, pG, GV)
    elif LOSS_PRED == 'mae':
        return loss_pred_mae_torch(pVv, pViv, pG, GV)
    elif LOSS_PRED == 'huber':
        return loss_pred_huber_torch(pVv, pViv, pG, GV)
    elif LOSS_PRED == 'crossEntropy_weighted':
        return loss_pred_crossEntropy_weighted_torch(pVv, pViv, pG, GV)
    elif LOSS_PRED == 'focal':
        return loss_pred_focal_torch(pVv, pViv, pG, GV)
    else:
        raise ValueError(f"Unknown loss function: {LOSS_PRED}")


def loss_reg_rmse_torch(p_hat, p_true, indices):
    """
    PyTorch version of RMSE regularization loss.
    """
    if len(indices) == 0:
        return torch.tensor(0.0)
    
    errors = []
    for idx in indices:
        model_idx = MODEL_ENUM[idx]
        if isinstance(p_true, dict):
            true_val = p_true[idx]
        else:
            true_val = torch.tensor(p_true[idx], dtype=torch.float32)
        errors.append(torch.square(p_hat[model_idx] - true_val))
    
    if not errors:
        return torch.tensor(0.0)
    
    return torch.sqrt(torch.mean(torch.stack(errors)))


def loss_reg_mae_torch(p_hat, p_true, indices):
    """
    PyTorch version of MAE regularization loss.
    """
    if len(indices) == 0:
        return torch.tensor(0.0)
    
    errors = []
    for idx in indices:
        model_idx = MODEL_ENUM[idx]
        if isinstance(p_true, dict):
            true_val = p_true[idx]
        else:
            true_val = torch.tensor(p_true[idx], dtype=torch.float32)
        errors.append(torch.abs(p_hat[model_idx] - true_val))
    
    if not errors:
        return torch.tensor(0.0)
        
    return torch.mean(torch.stack(errors))


def loss_reg_huber_torch(p_hat, p_true, indices):
    """
    PyTorch version of Huber regularization loss.
    """
    if len(indices) == 0:
        return torch.tensor(0.0)
    
    delta = 0.05
    errors = []
    for idx in indices:
        model_idx = MODEL_ENUM[idx]
        if isinstance(p_true, dict):
            true_val = p_true[idx]
        else:
            true_val = torch.tensor(p_true[idx], dtype=torch.float32)
        errors.append(p_hat[model_idx] - true_val)
    
    if not errors:
        return torch.tensor(0.0)
        
    errors_tensor = torch.stack(errors)
    return torch.mean(torch.where(torch.abs(errors_tensor) < delta,
                                  0.5 * errors_tensor**2,
                                  delta * (torch.abs(errors_tensor) - 0.5 * delta)))


def loss_reg_torch(p_hat, p_true, indices):
    """
    PyTorch version of regularization loss dispatcher.
    """
    if LOSS_REG == 'rmse':
        return loss_reg_rmse_torch(p_hat, p_true, indices)
    elif LOSS_REG == 'mae':
        return loss_reg_mae_torch(p_hat, p_true, indices)
    elif LOSS_REG == 'huber':
        return loss_reg_huber_torch(p_hat, p_true, indices)
    else:
        raise ValueError(f"Unknown regularization loss function: {LOSS_REG}")


def total_loss_torch(pVv, pViv, pG, GV, pVva, pViva, pGa, idV, idG, w):
    """
    PyTorch version of total loss function.
    
    Parameters:
    -----------
    pVv : torch.Tensor
        Validator valid probabilities
    pViv : torch.Tensor  
        Validator invalid probabilities
    pG : torch.Tensor
        Generator probabilities
    GV : torch.Tensor
        Ground truth GV matrix
    pVva : dict
        Ground truth validator valid probabilities
    pViva : dict
        Ground truth validator invalid probabilities  
    pGa : dict
        Ground truth generator probabilities
    idV : list
        Validator indices for regularization
    idG : list
        Generator indices for regularization
    w : list
        Loss weights [w_pred, w_gen_reg, w_val_reg1, w_val_reg2]
    """
    # Prediction loss
    l1 = loss_pred_torch(pVv, pViv, pG, GV)
    
    # Regularization losses
    l2 = loss_reg_torch(pG, pGa, idG)
    l3_1 = loss_reg_torch(pVv, pVva, idV)
    l3_2 = loss_reg_torch(pViv, pViva, idV)

    return l1 + w[0] * l2 + w[1] * l3_1 + w[2] * l3_2
