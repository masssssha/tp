import matplotlib.pyplot as plt
import numpy as np
from tsv_to_array import *
import numpy as np
import os.path
import sys

func_lst = ['MAMSE', 'MASSIM', 'MASTRESS', 'MANRMSD']
mask_lst = ['circle', 'square', 'crop']
folder = f'{os.path.dirname(os.path.dirname(sys.argv[0]))}/results'
if not os.path.isdir(folder):
    os.mkdir(folder)


def vizualize(x: list, y: list, z: list, label_1: str, label_2: str, filename: str,  xlabel: str, 
              ylabel: str, xticks: list = None, yticks: list = None) -> None:
    """Create plot for 2 datasets: y, z.

    Parameters:
        x : list
            Data for x-axis. This set is the same for y and z.
        y : list
            Data for y-axis.
        z : list
            Data for y-axis.
        label_1 : str
            Name for first dataset (y).
        label_2 : str
            Name for second dataset (z).
        filename : str
            Title for the plot.
        xlabel : str
            Name for x-axis.
        ylabel : str
            Name for y-axis.
        xticks : list
            Ticks for x-axis.
        yticks : list
            Ticks for y-axis.
        """
    plt.figure()
    plt.plot(x, y, label=label_1)
    plt.plot(x, z, label=label_2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(filename)
    plt.xlim(x[0], x[-1])
    if xticks is not None:
        plt.xticks(xticks)
    if yticks is not None:
        plt.yticks(yticks)
        plt.ylim(yticks[0], yticks[-1])
    plt.grid()
    plt.legend()
    plt.savefig(f'{folder}/{filename}')

def plot_violin(data: list, filename: str, labels: list) -> None:
    """Create violin plot."""
    plt.figure()
    plt.violinplot(data, showmedians=True)
    plt.title(filename)
    plt.xticks([i+1 for i in range(len(data))], labels=labels)
    plt.ylabel('SSIM')
    plt.savefig(filename)

def main():
    for i in range(len(func_lst)):
        for j in range(len(mask_lst)):
            file = f'{func_lst[i]}_{mask_lst[j]}_avg.tsv'
            rad, low, clinical = tsv_to_arr(folder + '/' + file)
            x_label = 'wing'
            if mask_lst[j] == 'circle':
                x_label = 'radius'
            vizualize(rad, low, clinical, 'low', 'clinical', f'{func_lst[i]}_{mask_lst[j]}', 
                      x_label, func_lst[i], 
                    np.arange(rad[0], rad[-1], 10))
        
if __name__ == "__main__":
    main()