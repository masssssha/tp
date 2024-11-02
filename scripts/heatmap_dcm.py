#Create heatmaps to show data range
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm
import pydicom
import os
import sys
import csv

def dcm_to_npy(path: str) -> np.ndarray:
    """Return 2d np.array with data type np.float32.
    
    Parameters:
    path : str
        Absolute file name of dcm format.
    """
    dicom = pydicom.dcmread(path)
    image = dicom.pixel_array.astype(np.float32)
    return image

def vizualize(img: np.ndarray, filename: str, range: list) -> None:
    """Show image of input array

    Parameters:
    img : np.ndarray
        Input data.

    filename : str
        Title for image and name for file.

    range : list
        Minimum and maximum values ​​for show an image.
    """
    plt.figure()
    plt.imshow(img, vmin=range[0], vmax=range[1], cmap=matplotlib.cm.afmhot)
    plt.colorbar()
    plt.title(filename)
    plt.savefig(f'{os.path.dirname(os.path.dirname(sys.argv[0]))}/images/{filename}.jpg')

def main():
    folder = os.path.dirname(os.path.dirname(sys.argv[0]))
    path_1 = folder + '/dcm/1-004.dcm'
    path_2 = folder + '/dcm/1-011.dcm'

    lidc_1 = dcm_to_npy(path_1)
    lidc_2 = dcm_to_npy(path_2)
    lidc_min = min(lidc_1.min(), lidc_2.min())
    lidc_max = max(lidc_1.max(), lidc_2.max())
    vizualize(lidc_1, 'LIDC-IDRI-0802_1-004', [lidc_min, lidc_max])
    vizualize(lidc_2, 'LIDC-IDRI-0807_1-011', [lidc_min, lidc_max])

    images = sorted(os.listdir(folder + '/npy'))
    table = [[0 for j in range(3)] for i in range(len(images)+1)]
    table[0] = ['id', 'min', 'max']
    for i in range(len(images)):
        table[i+1][0] = images[i]
        table[i+1][1] = (np.load(folder + '/npy/' + images[i])).min()
        table[i+1][2] = (np.load(folder + '/npy/' + images[i])).max()

    with open('npy_range.tsv', 'w') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerows(table)

if __name__ == "__main__":
    main()