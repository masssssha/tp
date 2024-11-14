import matplotlib.pyplot as plt
import numpy as np
import csv
import os.path
import sys

def tsv_to_arr(file: str) -> tuple[list, list, list]:
    """
    Return lists: radius, low_dose, clinical_dose based on data from file(.tsv).

    Parameters:
    file : str
        Absolute file name of tsv format.
    """
    with open(file, 'r', newline='\n') as file:
        reader = csv.reader(file, delimiter='\t')
        data = [row for row in reader]

    radius = []
    low_dose = []
    clinical_dose = []

    for i in range(1, len(data)):
        radius.append(np.float32(data[i][0]))
        low_dose.append(np.float32(data[i][1]))
        clinical_dose.append(np.float32(data[i][2]))

    return radius, low_dose, clinical_dose

def create_plot(x: list, y: list, z: list, label_1: str, label_2: str, filename: str,  xlabel: str, 
              ylabel: str, xticks: list, yticks: list) -> None:
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
    plt.savefig(f'{os.path.dirname(os.path.dirname(sys.argv[0]))}/images/{filename}')
    plt.show()

def main():
    radius, low, clinical = tsv_to_arr('C:/Users/mehov/tp/Average_MAMSE.tsv')
    minimum_y = round(min(min(low), min(clinical)), 2)
    maximum_y = round(max(max(low), max(clinical)), 2)
    create_plot(radius, low, clinical, 'low', 'clinical', 'MAMSE (radius)', 'radius', 'MAMSE',
                np.arange(radius[0], radius[-1], 10), np.arange(minimum_y, maximum_y, 0.01))

if __name__ == "__main__":
    main()