import os
from tsv_to_array import tsv_to_arr
import csv
import sys

folder = f'{os.path.dirname(os.path.dirname(sys.argv[0]))}/results'
func_lst = ['MAMSE', 'MASSIM', 'MASTRESS', 'MANRMSD']
kind_lst = ['circle', 'square', 'crop']

def save(table: list[list], filename: str) -> None:
    with open(f'results/{filename}.tsv', 'w') as file:
            writer = csv.writer(file, delimiter='\t')
            writer.writerows(table)

def calc_avg(folder: str, filename: str) -> None:
    """Calculate average value from tsv.
    
    Parameters:
        folder : str
            Folder where tsv are contained.
        filename: str
            File name."""
    content = sorted(os.listdir(folder))
    f = folder + '/' + content[0]
    rad, l, c = tsv_to_arr(f)
    avg_square = [[0 for j in range(3)] for i in range(len(rad)+1)]
    avg_square[0] = ['radius', 'low', 'clinical']

    for i in range(len(content)):
        file = folder + '/' + content[i]
        radius, low, clinical = tsv_to_arr(file)
        for j in range(len(radius)):
            avg_square[j+1][0] = radius[j]
            avg_square[j+1][1] += low[j]/len(content)
            avg_square[j+1][2] += clinical[j]/len(content)

    save(avg_square, filename)

def main():
    for i in range(len(func_lst)):
        for j in range(len(kind_lst)):
            calc_avg(f'{folder}/{func_lst[i]}_{kind_lst[j]}', f'{func_lst[i]}_{kind_lst[j]}_avg')
    
if __name__ == "__main__":
    main()