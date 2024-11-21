import numpy as np
import csv
import matplotlib.pyplot as plt
import cv2
import time
import os
import torch
from pytorch_ssim import ssim

folder_with_data = '/home/masha/validate.vol'
data_id = '/home/masha/number.txt'
path_to_save = '/home/masha/tp/results_3d'
header = ['radius', 'low', 'clinical']
rad_circle_start = 100
rad_circle_end = 182
radius_circle = [i for i in range(rad_circle_start, rad_circle_end + 1, 2)]
rad_square_start = 70
rad_square_end = 128
radius_square = [i for i in range(rad_square_start, rad_square_end + 1, 2)]
func_list = ['MASSIM']

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
        phantom: np.ndarray, 3-dimensional
        Correct values.

        recon: np.ndarray, 3-dimensional
        Estimated values.

        mask: np.ndarray, 3-d or 2-d
        Mask by which the MAMSE should be calculated. If None, a new mask will be created according to np.shape(phantom).
    Returns:
        value: np.float32 
        Masked mean squared error (the best value is 0.0).
    """
    if mask is None:
        mask = np.ones(np.shape(phantom))
    if mask.ndim == 2:
        mask = np.ones(np.shape(phantom))*mask
    if np.shape(phantom) != np.shape(recon) or np.shape(phantom) != np.shape(mask) or np.ndim(phantom) != 3:
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
    temp = np.zeros((np.shape(img)[0], 1, np.shape(img)[1], np.shape(img)[2]))
    temp[:, 0, :, :] = img.copy()
    return torch.Tensor(np.array(temp))


def MASSIM(phantom: np.ndarray, recon: np.ndarray, mask: np.ndarray = None):
    """Calculate Masked Structural Similarity Index Measure.
    
    Parameters:
        phantom: np.ndarray, 3-dimensional
        Correct values.

        recon: np.ndarray, 3-dimensional
        Estimated values.

        mask: np.ndarray, 3-d or 2-d
        Mask by which the MASSIM should be calculated. If None, a new mask will be created according to np.shape(phantom).
    Returns:
        value: np.float32 
        Masked Structural Similarity Index Measure (the best value is 1.0).
    """
    if mask is None:
        mask = np.ones(np.shape(phantom))
    if mask.ndim == 2:
        mask = np.ones(np.shape(phantom))*mask
    if np.shape(phantom) != np.shape(recon) or np.shape(phantom) != np.shape(mask) or np.ndim(phantom) != 3:
        raise ImError(np.shape(phantom), np.shape(recon), np.shape(mask))
    img_1 = phantom.copy()
    img_2 = recon.copy()
    mask_copy = mask.copy()

    img_1 = make_tensor(img_1)
    img_2 = make_tensor(img_2)
    mask_copy = make_tensor(mask_copy)
    mask_copy = mask_copy.bool()
    value = ssim(img_1, img_2, 1.0, mask=mask_copy)
    return value.item()

def MASTRESS(recon: np.ndarray, phantom: np.ndarray, mask: np.ndarray = None):
    """Calculate Masked standardised residual sum of squares.
    
    Parameters:
        phantom: np.ndarray, 3-dimensional
        Correct values.

        recon: np.ndarray, 3-dimensional
        Estimated values.

        mask: np.ndarray, 3-d or 2-d
        Mask by which the MASTRESS should be calculated. If None, a new mask will be created according to np.shape(phantom).
    Returns:
        value: np.float32 
        Masked standardised residual sum of squares.
    """
    if mask is None:
        mask = np.ones(np.shape(phantom))
    if mask.ndim == 2:
        mask = np.ones(np.shape(phantom))*mask
    if np.shape(phantom) != np.shape(recon) or np.shape(phantom) != np.shape(mask) or np.ndim(phantom) != 3:
        raise ImError(np.shape(phantom), np.shape(recon), np.shape(mask))
    
    img_1 = phantom.copy()
    img_2 = recon.copy()
    img_1[mask == 0] = 0
    img_2[mask == 0] = 0
    dot_prod = np.sum((img_1 * img_2))
    recon_l2 = np.sum(img_2 ** 2)
    phantom_l2 = np.sum(img_1 ** 2)
    result = dot_prod / (recon_l2 * phantom_l2) ** 0.5
    result *= result
    result = 1 - result
    result = result ** 0.5
    return result

def MANRMSD(recon: np.ndarray, phantom: np.ndarray, mask: np.ndarray = None):
    """Calculate Masked normalized root mean squared difference.
    
    Parameters:
        phantom: np.ndarray, 3-dimensional
        Correct values.

        recon: np.ndarray, 3-dimensional
        Estimated values.

        mask: np.ndarray, 3-d or 2-d
        Mask by which the MANRMSD should be calculated. If None, a new mask will be created according to np.shape(phantom).
    Returns:
        value: np.float32 
        Masked normalized root mean squared difference.
    """
    if mask is None:
        mask = np.ones(np.shape(phantom))
    if mask.ndim == 2:
        mask = np.ones(np.shape(phantom))*mask
    if np.shape(phantom) != np.shape(recon) or np.shape(phantom) != np.shape(mask) or np.ndim(phantom) != 3:
        raise ImError(np.shape(phantom), np.shape(recon), np.shape(mask))
    
    img_1 = phantom.copy()
    img_2 = recon.copy()
    img_1[mask == 0] = 0
    img_2[mask == 0] = 0
    diff_sqr = np.zeros((recon.shape[0], recon.shape[1]))
    diff_sqr = (img_2 - img_1) ** 2
    nrmsd = (np.sum(diff_sqr) / np.sum(img_1 ** 2)) ** 0.5
    return nrmsd

func_dict = {'MAMSE': MAMSE, 'MASSIM': MASSIM, 'MASTRESS': MASTRESS, 'MANRMSD': MANRMSD}

def create_circle_mask(radius: int) -> np.ndarray:
    """Create circle mask by radius."""
    mask = cv2.circle(np.zeros((256, 256)), (128, 128), radius, (1), -1)
    return mask

def create_square_mask(radius: int) -> np.ndarray:
    """Create square mask by half the side of a square."""
    mask = cv2.rectangle(np.zeros((256, 256)), (128-radius, 128-radius), 
                         (127+radius, 127+radius), (1), -1)
    return mask

def create_crop(ref: np.ndarray, radius: int) -> np.ndarray:
    """Create creates a square cutout from an image corresponding to half the side of a square."""
    img = ref.copy()
    return img[:, (128-radius):(128+radius), (128-radius):(128+radius)]

def save_table(table: list[list], filename: str, path_to_save: str = None) -> None:
    """Save table."""
    if path_to_save is None:
        with open(f'{filename}.tsv', 'w') as file:
            writer = csv.writer(file, delimiter='\t')
            writer.writerows(table)
    else:
        with open(f'{path_to_save}/{filename}.tsv', 'w') as file:
            writer = csv.writer(file, delimiter='\t')
            writer.writerows(table)

def calculate_statistics_circle(func_lst: list[str], vol_ref: np.ndarray, 
                        vol_low: np.ndarray, vol_clinical: np.ndarray, mask: list, id: int) -> None:
    tables = []
    for i in range(len(func_lst)):
        tab = [[0 for j in range(3)] for k in range(len(mask)+1)]
        tab[0] = header
        tables.append(tab)
    for rad_id in range(len(mask)):
        for i in range(len(func_lst)):
            if func_lst[i] == 'MASSIM':
                img_1 = normalized(vol_ref)
                img_2 = normalized(vol_low)
                img_3 = normalized(vol_clinical)
            else:
                img_1 = vol_ref
                img_2 = vol_low
                img_3 = vol_clinical
            tables[i][rad_id+1][0] = radius_circle[rad_id]
            tables[i][rad_id+1][1] += (func_dict[func_lst[i]](img_1, img_2, mask[rad_id]))
            tables[i][rad_id+1][2] += (func_dict[func_lst[i]](img_1, img_3, mask[rad_id]))

    for i in range(len(func_lst)):
        if not os.path.isdir(f'{path_to_save}/{func_lst[i]}_circle'):
            os.mkdir(f'{path_to_save}/{func_lst[i]}_circle')
        save_table(tables[i], f'{path_to_save}/{func_lst[i]}_circle/{func_lst[i]}_circle_{id}')

def calculate_statistics_square_crop(func_lst: list[str], vol_ref: np.ndarray, 
                        vol_low: np.ndarray, vol_clinical: np.ndarray, mask: list, id: int) -> None:
    tables_square = []
    tables_crop = []
    for i in range(len(func_lst)):
        tab = [[0 for j in range(3)] for k in range(len(mask)+1)]
        tab[0] = header
        tables_square.append(tab)
        tab = [[0 for j in range(3)] for k in range(len(mask)+1)]
        tab[0] = header
        tables_crop.append(tab)
    for rad_id in range(len(mask)):
        for i in range(len(func_lst)):
            if func_lst[i] == 'MASSIM':
                img_1 = normalized(vol_ref)
                img_2 = normalized(vol_low)
                img_3 = normalized(vol_clinical)
            else:
                img_1 = vol_ref
                img_2 = vol_low
                img_3 = vol_clinical
            tables_square[i][rad_id+1][0] = radius_square[rad_id]
            tables_square[i][rad_id+1][1] += (func_dict[func_lst[i]](img_1, img_2, mask[rad_id]))
            tables_square[i][rad_id+1][2] += (func_dict[func_lst[i]](img_1, img_3, mask[rad_id]))
            crop_1 = create_crop(img_1, radius_square[rad_id])
            crop_2 = create_crop(img_2, radius_square[rad_id])
            crop_3 = create_crop(img_3, radius_square[rad_id])

            tables_crop[i][rad_id+1][0] = radius_square[rad_id]
            tables_crop[i][rad_id+1][1] += (func_dict[func_lst[i]](crop_1, crop_2))
            tables_crop[i][rad_id+1][2] += (func_dict[func_lst[i]](crop_1, crop_3))

    for i in range(len(func_lst)):
        if not os.path.isdir(f'{path_to_save}/{func_lst[i]}_square'):
            os.mkdir(f'{path_to_save}/{func_lst[i]}_square')
        save_table(tables_square[i], f'{path_to_save}/{func_lst[i]}_square/{func_lst[i]}_square_{id}')
        if not os.path.isdir(f'{path_to_save}/{func_lst[i]}_crop'):
            os.mkdir(f'{path_to_save}/{func_lst[i]}_crop')
        save_table(tables_crop[i], f'{path_to_save}/{func_lst[i]}_crop/{func_lst[i]}_crop_{id}')

def main():
    with open(data_id, 'r') as file:
        content = file.read().split('\n')

    for id in range(len(content)):
        print(f'Started {content[id]}', time.ctime())

        #load data
        clean_vol = np.load(f'{folder_with_data}/{content[id]}_clean_fdk_256.npy')
        fdk_low_dose = np.load(f'{folder_with_data}/{content[id]}_fdk_low_dose_256.npy')
        fdk_clinical_dose = np.load(f'{folder_with_data}/{content[id]}_fdk_clinical_dose_256.npy')

        #create masks
        circle_masks = []
        for j in range(rad_circle_start, rad_circle_end+1, 2):
            circle_masks.append(create_circle_mask(j))

        square_masks = []
        for j in range(rad_square_start, rad_square_end+1, 2):
            square_masks.append(create_square_mask(j))
        
        calculate_statistics_circle(func_list, clean_vol, fdk_low_dose, fdk_clinical_dose, circle_masks, 801+id)
        print('Calculated circle', time.ctime())
        calculate_statistics_square_crop(func_list, clean_vol, fdk_low_dose, fdk_clinical_dose, square_masks, 801+id)
        print(f'Finished {content[id]}', time.ctime())
    
if __name__ == "__main__":
    main()
