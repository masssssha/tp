import matplotlib.pyplot as plt
import numpy as np
import csv
from os.path import dirname
import sys
from tsv_to_array import tsv_to_arr

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

def main():
    folder = f'{dirname(dirname(sys.argv[0]))}/results/'
    filename = 'Average_MASSIM_square.tsv'
    radius, low, clinical = tsv_to_arr(folder+filename)
    minimum_y = round(min(min(low), min(clinical)), 1)
    maximum_y = round(max(max(low), max(clinical)), 1)
    create_plot(radius, low, clinical, 'low', 'clinical', 'MASSIM_circle (radius)', 'radius', 'MASSIM',
                np.arange(radius[0], radius[-1], 10), np.arange(minimum_y, maximum_y, 0.1), 
                f'{dirname(dirname(sys.argv[0]))}/images')

if __name__ == "__main__":
    main()