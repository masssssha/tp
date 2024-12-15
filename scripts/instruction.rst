How to work with scripts?
==============================

The :ref:`metrics_ref` contains evaluation functions: MAMSE, MASSIM, MASTRESS, MANRMSD — and functions for calculating these statistics: calculate_statistics_circle, calculate_statistics_square_crop. 
To call functions on your data, you need to change the variables *folder_with_data* - folder with volumes, *data_id* - the txt file with package numbers, *results_folder* - the folder where the calculation results will be saved. After that just run the script.


calc_avg_tabs.py calculates the average value of statistics by packages. You need to change the value of the *folder* - folder with the results of calculating all statistics (it was *results_folder*) to yours. After that just run the script (after metrics.py). The results of calculating average values ​​will be saved in the *tp/results* automatically.


The :ref:`second` by average values ​​takes as input: a list of radii (which is plotted along the x-axis), a list of values ​​for low_dose (which is plotted along the y-axis), a list of values ​​for clinical_dose (which is plotted along the y-axis), label_1 for the low_dose label, label_2 for the clinical_dose label, a title for the graph (which is also the name under which the file will be saved), labels for the x- and y-axes, and divisions along the x- and y-axes (optional) – plots the graph and saves it.
Just run the script (after calculate_average.py). The plots ​​will be saved in the *tp/results* automatically.