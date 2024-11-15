import os
from tsv_to_array import tsv_to_arr
from metrics import save_table

def create_avg_from_table(folder: str, rows: int, cols: int, header: list, filename: str,
                          folder_to_save: str):
    content = sorted(os.listdir(folder))
    avg = [[0 for j in range(cols)] for i in range(rows)]
    avg[0] = header
    for i in range(len(content)):
        rad, low, clinical = tsv_to_arr(f'{folder}/{content[i]}')
        for j in range(len(rad)):
            avg[j+1][0] = rad[j]
            avg[j+1][1] += low[j]/len(content)
            avg[j+1][2] += clinical[j]/len(content)
    save_table(avg, filename, folder_to_save)

def main():
    create_avg_from_table('/home/masha/tp/results/MASSIM_crop', 43, 3, ['radius', 'low', 'clinical'], 
                          'Average_MASSIM_crop', '/home/masha/tp/results')

if __name__ == "__main__":
    main()