import matplotlib.pyplot as plt
import numpy as np
from os.path import dirname
import sys

def vizualize(img: np.ndarray, filename: str, save: bool = True, show: bool = True, path_to_save: str = None):
    """Parameters:
        img: np.ndarray, shape(n, m)
        filename: str
    """
    plt.figure()
    plt.imshow(img)
    plt.title(filename)
    plt.colorbar()
    if save:
        if path_to_save is None:
            plt.savefig(filename)
        else:
            plt.savefig(path_to_save + '/' + filename)
    if show:
        plt.show()