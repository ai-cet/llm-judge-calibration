from src.regression.gradient_descent.torch_loss import total_loss_torch
from src.regression.config import COLOR_GREEN_DELTA, COLOR_YELLOW_DELTA, COMPUTE_REFERENCE_VALUES, ENSEMBLE_MAJORITY_MAXERR, ENSEMBLE_MAJORITY_MEANERR, ENSEMBLE_BEST_MAXERR, ENSEMBLE_BEST_MEANERR, K_LIST, LLM_VALIDATOR_MAXERR_RANGE, LLM_VALIDATOR_MEANERR_RANGE, LOSS_PRED, NUM_RUNS, MAX_WORKERS, WEIGHTS
from src.regression.config import PV_START, PVIV_START, PG_START, ERR_EPSILON_PG, ERR_EPSILON_PIV, ERR_EPSILON_PV, GENS, MODELS, MODEL_NAMES, MODEL_ENUM
from src.regression.config import VALIDATOR_COUNTS_CONST, pGa_CONST, GV_CONST, PVVA_CONST, PVIVA_CONST
from src.regression.config import PATH_LOGS, PATH_LATEX, PATH_REGRESSION, PATH_LOGS_LOSS_ITER, fpathRegressionSummary
from src.config import VALIDATOR_REPAIR_SUFFIX

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.optimize import minimize
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed

torch.manual_seed(42)
scipy_seed = 42

# Set device for PyTorch
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')


####################
# PyTorch Model
####################

class RegressionModel(nn.Module):
    """
    PyTorch model for regression optimization.
    """
    def __init__(self, num_validators, num_generators, pVv_init=None, pViv_init=None, pG_init=None):
        super(RegressionModel, self).__init__()
        
        # Initialize with provided values or defaults (no logit conversion)
        if pVv_init is not None:
            self.pVv = nn.Parameter(torch.tensor(pVv_init, dtype=torch.float32))
        else:
            self.pVv = nn.Parameter(torch.rand(num_validators))
            
        if pViv_init is not None:
            self.pViv = nn.Parameter(torch.tensor(pViv_init, dtype=torch.float32))
        else:
            self.pViv = nn.Parameter(torch.rand(num_validators))
            
        if pG_init is not None:
            self.pG = nn.Parameter(torch.tensor(pG_init, dtype=torch.float32))
        else:
            self.pG = nn.Parameter(torch.rand(num_generators))
        
        # Move model to device
        self.to(device)
        
    def forward(self):
        # Return parameters directly without sigmoid activation
        return self.pVv, self.pViv, self.pG

# =========================
#   Estimate for a given k
# =========================

def estimate_probs_grad_descent(k, GV, pVva, pViva, pGa, idV, idG, w = [1, 0, 0]):
    final_loss_best = np.inf

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

    # Convert to PyTorch tensors and move to device
    GV_tensor = torch.tensor(GV, dtype=torch.float32).to(device)
    pVva_tensor = {k: torch.tensor(v, dtype=torch.float32).to(device) for k, v in pVva.items()} if pVva else {}
    pViva_tensor = {k: torch.tensor(v, dtype=torch.float32).to(device) for k, v in pViva.items()} if pViva else {}
    pGa_tensor = {k: torch.tensor(v, dtype=torch.float32).to(device) for k, v in pGa.items()}

    for run_count in range(NUM_RUNS):
        if run_count != 0:
            pG = pG_ + np.random.uniform(-ERR_EPSILON_PG/2, ERR_EPSILON_PG/2, NUM_GENERATORS)
            pVv_hat = pVv_hat_ + np.random.uniform(-ERR_EPSILON_PV/2, ERR_EPSILON_PV/2, NUM_VALIDATORS)
            pViv_hat = pViv_hat_ + np.random.uniform(-ERR_EPSILON_PIV/2, ERR_EPSILON_PIV/2, NUM_VALIDATORS)
        else:
            pG, pVv_hat, pViv_hat = pG_, pVv_hat_, pViv_hat_

        pG = np.clip(pG, 0, 1)
        pVv_hat = np.clip(pVv_hat, 0, 1)
    pViv_hat = np.clip(pViv_hat, 0, 1)  

    # Create PyTorch model with initialization values
    model = RegressionModel(NUM_VALIDATORS, NUM_GENERATORS, 
                            pVv_init=pVv_hat, pViv_init=pViv_hat, pG_init=pG)

    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    # Setup learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, 
                                                    patience=200, verbose=True, min_lr=1e-7)
    
    # Training loop
    num_epochs = 100000
    best_loss = float('inf')
    patience = 1000
    patience_counter = 0
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        pVv_pred, pViv_pred, pG_pred = model()
        
        loss = total_loss_torch(pVv_pred, pViv_pred, pG_pred, GV_tensor, 
                                pVva_tensor, pViva_tensor, pGa_tensor, idV, idG, w)
        
        loss.backward()
        optimizer.step()
        
        # Step the scheduler
        scheduler.step(loss.item())
        
        # Early stopping
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
            # Save best parameters
            best_pVv = pVv_pred.detach().clone()
            best_pViv = pViv_pred.detach().clone()
            best_pG = pG_pred.detach().clone()
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
            
    print(f"Run {run_count} converged with loss: {best_loss:.6f}")
    
    # Convert back to numpy for compatibility with existing code
    pVv = best_pVv.detach().cpu().numpy()
    pViv = best_pViv.detach().cpu().numpy()
    pG = best_pG.detach().cpu().numpy()
    
    # Update best results
    if best_loss < final_loss_best:
        final_loss_best = best_loss
        final_pVv = pVv
        final_pViv = pViv 
        final_pG = pG
    
    return final_pVv, final_pViv, final_pG