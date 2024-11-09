import numpy as np
from sklearn.metrics import mean_squared_error
import csv
import matplotlib.pyplot as plt
import cv2
import time

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

def create_circle_mask(radius: int) -> np.ndarray:
    mask = cv2.circle(np.zeros((256, 256)), (128, 128), radius, (1), -1)
    return mask

def create_square_mask(radius: int) -> np.ndarray:
    mask = cv2.rectangle(np.zeros((256, 256)), (128-radius, 128-radius), 
                         (127+radius, 127+radius), (1), -1)
    return mask

def create_crop(ref: np.ndarray, radius: int) -> np.ndarray:
    img = ref.copy()
    return img[(128-radius):(128+radius), (128-radius):(128+radius)]

def save_table(table: list[list], filename: str) -> None:
    with open(f'{filename}.tsv', 'w') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerows(table)

def main():
    folder = '/home/masha/validate.vol'
    file_txt = '/home/masha/number.txt'
    with open(file_txt, 'r') as file:
        content = file.read().split('\n')
    header = ['radius', 'low', 'clinical']

    avg_mse = [[0 for j in range(3)] for i in range(100, 184+1, 2)]
    avg_mse[0] = header

    for id in range(len(content)):
        print(f'Started {content[id]}', time.ctime())

        #create tables
        tab_mse_circle = [[0 for j in range(3)] for i in range(100, 184+1, 2)]
        tab_mse_circle[0] = header

        #load data
        clean_vol = np.load(f'{folder}/{content[id]}_clean_fdk_256.npy')
        fdk_low_dose = np.load(f'{folder}/{content[id]}_fdk_low_dose_256.npy')
        fdk_clinical_dose = np.load(f'{folder}/{content[id]}_fdk_clinical_dose_256.npy')

        #create masks
        circle_masks = []
        for j in range(100, 182+1, 2):
            circle_masks.append(create_circle_mask(j))

        square_masks = []
        for j in range(70, 128+1, 2):
            square_masks.append(create_square_mask(j))
        
        #calculate statistics for rad. across all slices
        for count in range(len(circle_masks)):
            mse_low, mse_clinical = 0, 0
            for n_slice in range(np.shape(clean_vol)[0]):
                mse_low += MAMSE(clean_vol[n_slice], fdk_low_dose[n_slice], circle_masks[count])
                mse_clinical += MAMSE(clean_vol[n_slice], fdk_clinical_dose[n_slice], circle_masks[count])

            tab_mse_circle[count+1][0] = 100 + 2*count
            tab_mse_circle[count+1][1] = mse_low/np.shape(clean_vol)[0]
            tab_mse_circle[count+1][2] = mse_clinical/np.shape(clean_vol)[0]

            avg_mse[count+1][0] = 100 + 2*count
            avg_mse[count+1][1] += mse_low/(np.shape(clean_vol)[0] * len(content))
            avg_mse[count+1][2] = mse_clinical/(np.shape(clean_vol)[0] * len(content))

            print(f'Finished radius {100+2*count}', time.ctime())
        save_table(tab_mse_circle, f'/home/masha/results/mamse/MAMSE_{content[id]}')
        print(f'Finished {content[id]}', time.ctime())
    save_table(avg_mse, '/home/masha/results/Average_MAMSE')
    
if __name__ == "__main__":
    main()