# Repository for a programming technology project
scripts - script folder

images - folder with plots/heatmaps

results - folder with results of calculation of functions (.tsv)

dcm - folder with .dcm

npy - folder with .npy

## Brief description of py files:
### ct-iqa/metrics.py
It contains evaluation functions: MAMSE, MASSIM, MASTRESS, MANRMSD — and functions for calculating these statistics: calculate_statistics_circle, calculate_statistics_square_crop. 
To call functions on your data, you need to change the variables *folder_with_data* - folder with volumes, *data_id* - the txt file with package numbers, *results_folder* - the folder where the calculation results will be saved. After that just run the script.
### ct-iqa/calculate_average.py
Calculates the average value of statistics by packages. You need to change the value of the *folder* - folder with the results of calculating all statistics (it was *results_folder*) to yours. After that just run the script (after metrics.py). The results of calculating average values ​​will be saved in the *for-tomo-se/results* automatically.
### ct-iqa/plot.py
Just run the script (after calculate_average.py). The plots ​​will be saved in the *for-tomo-se/results* automatically.
