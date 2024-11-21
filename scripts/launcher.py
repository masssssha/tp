import matplotlib.pyplot as plt
import numpy as np
import csv
import os
import sys
from tsv_to_array import tsv_to_arr
from vizualize import *

def main():
    '''folder = f'{dirname(dirname(sys.argv[0]))}/results/'
    filename = 'Average_MASSIM_square.tsv'
    radius, low, clinical = tsv_to_arr(folder+filename)
    minimum_y = round(min(min(low), min(clinical)), 1)
    maximum_y = round(max(max(low), max(clinical)), 1)
    create_plot(radius, low, clinical, 'low', 'clinical', 'MASSIM_circle (radius)', 'radius', 'MASSIM',
                np.arange(radius[0], radius[-1], 10), np.arange(minimum_y, maximum_y, 0.1), 
                f'{dirname(dirname(sys.argv[0]))}/images')'''
    folder_2 = f'{dirname(dirname(sys.argv[0]))}/results/MANRMSD_circle'
    folder_3 = f'{dirname(dirname(sys.argv[0]))}/results_3d/MANRMSD_circle'
    content_2 = sorted(os.listdir(folder_2))
    content_3 = sorted(os.listdir(folder_3))
    diff_low = []
    diff_cl = []
    for i in range(len(content_2)):
        rad, low_square, cl_square = tsv_to_arr(folder_2 + '/' + content_2[i])
        rad, low_crop, cl_crop = tsv_to_arr(folder_3 + '/' + content_3[i])
        for j in range(len(rad)):
            diff_low.append(low_square[j] - low_crop[j])
            diff_cl.append(cl_square[j] - cl_crop[j])
            '''if cl_square[j] - cl_crop[j] == np.float32(-0.01404351):
                print(low_square[j], low_crop[j], cl_square[j], cl_crop[j])'''
    print(min(diff_low), max(diff_low), min(diff_cl), max(diff_cl))
    '''avg_square = f'{dirname(dirname(sys.argv[0]))}/results/Average_MANRMSD_circle.tsv'
    #avg_crop = f'{dirname(dirname(sys.argv[0]))}/results/Average_MASSIM_crop.tsv'
    diff_low = []
    diff_cl = []
    rad, low_square, cl_square = tsv_to_arr(avg_square)
    #rad, low_crop, cl_crop = tsv_to_arr(avg_crop)
    for j in range(len(rad)):
        diff_low.append(low_square[j])
        diff_cl.append(cl_square[j])
    min_y = round(min(min(diff_low), min(diff_cl)), 3)
    max_y = round(max(max(diff_low), max(diff_cl)), 3)
    create_plot(rad, diff_low, diff_cl, 'low', 'clinical', 'MANRMSD_circle(radius)', 
                'radius', 'MANRMSD', np.arange(rad[0], rad[-1], 10), np.arange(min_y, max_y, 0.1), 
                f'{dirname(dirname(sys.argv[0]))}/images')'''

if __name__ == "__main__":
    main()