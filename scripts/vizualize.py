import matplotlib.pyplot as plt
import numpy as np
from os.path import dirname
import sys

def vizualize(img: np.ndarray, filename: str, save: bool = True, show: bool = True):
    """Parameters:
        img: np.ndarray, shape(n, m)
        filename: str
    """
    plt.figure()
    plt.imshow(img)
    plt.title(filename)
    plt.colorbar()
    if save:
        plt.savefig(dirname(dirname(sys.argv[0])) + '/images/' + filename)
    if show:
        plt.show()