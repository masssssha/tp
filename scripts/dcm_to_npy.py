import pydicom
import os
import csv
import numpy as np
import time

path_dir = '/home/masha/LIDC-IDRI.validate/'
dirs = sorted(os.listdir(path_dir))
dir_set = []
data = [[0 for j in range(8)] for i in range(len(os.listdir(path_dir))+1)] #create a table for .tsv
data[0] = ['id', 'slice_n', 'thickness', 'vox_neg', 'min', 'index(min)', 'max', 'index(max)']

for i in range(len(dirs)):
    name = dirs[i].split('-')
    data[i+1][0] = name[2] #fill in the first column with file id
    for subdirs in os.listdir(path_dir+dirs[i]):
        for subdirs_2 in os.listdir(path_dir+dirs[i]+'/'+subdirs):
            dir_set.append(f'{path_dir}{dirs[i]}/{subdirs}/{subdirs_2}/') #get folders with data

for i in range(len(dir_set)):
    dirname = dir_set[i]
    dicom_set = sorted(os.listdir(dirname)) #get .dcm files from folders
    n_slices = len(dicom_set) #get number of slices
    slice_thickness = 0
    count = 0 #number of voxels less than -1024
    minimum= [] #list of min value from each slice
    maximum = [] #list of max value from each slice
    for filename in dicom_set:
        dicom = pydicom.dcmread(dirname+filename)
        slice_thickness = dicom.SliceThickness
        vol = dicom.pixel_array.astype(np.float32) #get np.ndarray
        vol -= 1024 #shift values
        count += np.sum(vol < -1024)
        minimum.append(np.min(vol))
        maximum.append(np.max(vol))
    #fill in the table
    data[i+1][1] = n_slices
    data[i+1][2] = format(slice_thickness, '.2f')
    data[i+1][3] = (count/(n_slices*512*512))
    data[i+1][4] = np.min(minimum)
    data[i+1][5] = minimum.index(np.min(minimum))
    data[i+1][6] = np.max(maximum)
    data[i+1][7] = maximum.index(np.max(maximum))
    print(data[i+1][0], time.ctime())

with open('dcm_info.tsv', 'w') as file:
    writer = csv.writer(file, delimiter='\t')
    writer.writerows(data)
