import numpy as np
from sklearn.metrics import mean_squared_error
import csv
import matplotlib.pyplot as plt

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
        phantom: np.ndarray, shape (n, n)
        Correct values.

        recon: np.ndarray, shape (n, n)
        Estimated values.

        mask: np.ndarray, shape (n, n), data_range=(0.0, 1.0)

    Returns:
        value: np.float32 
        Masked mean squared error (the best value is 0.0).
    """
    if np.shape(phantom) != np.shape(mask) != np.shape(recon):
        raise ValueError(
            "Phantom, recon and mask must have the same shape ({0}, {1}, {2}).".format(
            np.shape(phantom), np.shape(recon), np.shape(mask)))
    img_1 = phantom.copy()
    img_2 = recon.copy()
    img_1[mask == 0] = 0
    img_2[mask == 0] = 0
    return np.sum((img_1 - img_2)**2)/np.sum(mask > 0)