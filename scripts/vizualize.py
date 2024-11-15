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

def create_plot(x: list, y: list, z: list, label_1: str, label_2: str, filename: str,  xlabel: str, 
              ylabel: str, xticks: list, yticks: list, path_to_save: str = None) -> None:
    plt.figure()
    plt.plot(x, y, label=label_1)
    plt.plot(x, z, label=label_2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(filename)
    plt.xlim(x[0], x[-1])
    plt.xticks(xticks)
    plt.yticks(yticks)
    plt.grid()
    plt.legend()
    if path_to_save is None:
        plt.savefig(filename)
    else:
        plt.savefig(f'{path_to_save}/{filename}')
    plt.show()