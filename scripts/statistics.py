import numpy as np
from sklearn.metrics import mean_squared_error
import csv
import matplotlib.pyplot as plt
from ind_by_name import *

def MSE(phantom: np.ndarray, recon: np.ndarray) -> np.float32:
    """Calculate mean squared error.

    Parameters:
        phantom: np.ndarray of shape (n, n)
        Correct values.

        recon: np.ndarray of shape (n, n)
        Estimated values.

    Returns:
        value: np.float32 
        Mean squared error (the best value is 0.0).
    """
    if np.shape(phantom) != np.shape(recon):
        raise ValueError(
            "Phantom and recon must have the same shape ({0} != {1}).".format(
            np.shape(phantom), np.shape(recon)))
    return np.average((phantom - recon)**2)

def MAMSE(phantom: np.ndarray, recon: np.ndarray, mask: np.ndarray) -> np.float32:
    """Calculate masked mean squared error.
    
    Parameters:
        phantom: np.ndarray of shape (n, n)
        Correct values.

        recon: np.ndarray of shape (n, n)
        Estimated values.

        mask: np.ndarray of shape (n, n)

    Returns:
        value: np.float32 
        Masked mean squared error (the best value is 0.0).
    """
    if np.shape(phantom) != np.shape(mask) != np.shape(recon):
        raise ValueError(
            "Phantom, recon and mask must have the same shape ({0}, {1}, {2}).".format(
            np.shape(phantom), np.shape(recon), np.shape(mask)))
    phantom[mask == 0] = 0
    recon[mask == 0] = 0
    return np.sum((phantom - recon)**2)/np.sum(mask > 0)