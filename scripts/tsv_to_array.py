import numpy as np
import csv

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
        radius.append(int(data[i][0]))
        low_dose.append(np.float32(data[i][1]))
        clinical_dose.append(np.float32(data[i][2]))

    return radius, low_dose, clinical_dose