import numpy as np
import csv
import matplotlib.pyplot as plt
import cv2
from time import ctime
import os
import torch
from pytorch_ssim import ssim

class ImError(Exception):
    def __init__(self, shape_1, shape_2, shape_3):
        self.shape_1 = shape_1
        self.shape_2 = shape_2
        self.shape_3 = shape_3
    def __str__(self):
        return f"Invalid image sizes: {self.shape_1}, {self.shape_2}, {self.shape_3}."
    
def MAMSE(phantom: np.ndarray, recon: np.ndarray, mask: np.ndarray = None) -> np.float32:
    """Calculate Masked Mean Squared Error.
    
    Parameters:
        phantom: np.ndarray, shape (n, n)
        Correct values.

        recon: np.ndarray, shape (n, n)
        Estimated values.

        mask: np.ndarray
        Mask by which the MAMSE should be calculated. If None, a new mask will be created according to np.shape(phantom).
    Returns:
        value: np.float32 
        Masked mean squared error (the best value is 0.0).
    """
    if mask is None:
        mask = np.ones(np.shape(phantom))
    if np.shape(phantom) != np.shape(recon) != np.shape(mask):
        raise ImError(np.shape(phantom), np.shape(recon), np.shape(mask))
    img_1 = phantom.copy()
    img_2 = recon.copy()
    img_1[mask == 0] = 0
    img_2[mask == 0] = 0
    return np.sum((img_1 - img_2)**2)/np.nonzero(mask)

def normalized(ref: np.ndarray, tr_low: np.float32 = -0.2, tr_hi: np.float32 = 3.5) -> np.ndarray:
    """Normalize image.
    
    Parameters:
        ref: np.ndarray
        Not normalized image.
    
    Returns:
        img: np.ndarray
        Normalized image.
    """
    img = ref.copy()
    img = (img - tr_low)/(tr_hi - tr_low)
    img = np.where(img < 0.0, 0.0, img)
    img = np.where(img > 1.0, 1.0, img)
    return img

def make_tensor(img: np.ndarray) -> torch.Tensor:
    """Make tensor from np.ndarray
    
    Parameters:
        img: np.ndarray
        The standard image.

    Returns:
        torch.Tensor
    """
    temp = np.zeros((1, 1, np.shape(img)[0], np.shape(img)[1]))
    temp[0, 0, :, :] = img.copy()
    return torch.Tensor(np.array(temp))


def MASSIM(phantom: np.ndarray, recon: np.ndarray, mask: np.ndarray = None):
    """Calculate Masked Structural Similarity Index Measure.
    
    Parameters:
        phantom: np.ndarray, shape (n, n)
        Correct values.

        recon: np.ndarray, shape (n, n)
        Estimated values.

        mask: np.ndarray
        Mask by which the MASSIM should be calculated. If None, a new mask will be created according to np.shape(phantom).
    Returns:
        value: np.float32 
        Masked Structural Similarity Index Measure (the best value is 1.0).
    """
    if mask is None:
        mask = np.ones(np.shape(phantom))
    if np.shape(phantom) != np.shape(recon) != np.shape(mask):
        raise ImError(np.shape(phantom), np.shape(recon), np.shape(mask))
    img_1 = phantom.copy()
    img_2 = recon.copy()
    mask_copy = mask.copy()

    img_1 = make_tensor(img_1)
    img_2 = make_tensor(img_2)
    mask_copy = make_tensor(mask_copy)
    mask_copy = mask_copy.bool()
    value = ssim(img_1, img_2, 1.0, mask=mask_copy)
    return value

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

    avg_ssim = [[0 for j in range(3)] for i in range(100, 184+1, 2)]
    avg_ssim[0] = header

    for id in range(len(content)):
        print(f'Started {content[id]}', ctime())

        #create tables
        tab_ssim_circle = [[0 for j in range(3)] for i in range(100, 184+1, 2)]
        tab_ssim_circle[0] = header

        #load data
        clean_vol = np.load(f'{folder}/{content[id]}_clean_fdk_256.npy')
        fdk_low_dose = np.load(f'{folder}/{content[id]}_fdk_low_dose_256.npy')
        fdk_clinical_dose = np.load(f'{folder}/{content[id]}_fdk_clinical_dose_256.npy')

        #create masks
        circle_masks = []
        for j in range(100, 182+1, 2):
            circle_masks.append(create_circle_mask(j))
        
        #calculate statistics for rad. across all slices
        for count in range(len(circle_masks)):
            ssim_low, ssim_clinical = 0, 0
            for n_slice in range(np.shape(clean_vol)[0]):
                ssim_low += MASSIM(clean_vol[n_slice], fdk_low_dose[n_slice], circle_masks[count])
                ssim_clinical += MASSIM(clean_vol[n_slice], fdk_clinical_dose[n_slice], circle_masks[count])

            tab_ssim_circle[count+1][0] = 100 + 2*count
            tab_ssim_circle[count+1][1] = ssim_low/np.shape(clean_vol)[0]
            tab_ssim_circle[count+1][2] = ssim_clinical/np.shape(clean_vol)[0]

            avg_ssim[count+1][0] = 100 + 2*count
            avg_ssim[count+1][1] += ssim_low/(np.shape(clean_vol)[0] * len(content))
            avg_ssim[count+1][2] += ssim_clinical/(np.shape(clean_vol)[0] * len(content))

            print(f'Finished radius {100+2*count}', ctime())
        if not os.path.isdir('results/MASSIM'):
            os.mkdir('results/MASSIM')
        save_table(tab_ssim_circle, f'/home/masha/tp/results/MASSIM/MASSIM_circle_{content[id]}')
        print(f'Finished {content[id]}', ctime())
    save_table(avg_ssim, '/home/masha/tp/results/Average_MASSIM')
    
if __name__ == "__main__":
    main()