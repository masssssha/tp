Formulas Used
=============

The following metrics were used in this project: MSE, SSIM, STRESS, NRMSD. A calculation for the mask was made for each of them.

MAMSE
"""""

The calculation was carried out according to this formula 
:math:`MAMSE = \frac{1}{n}\displaystyle\sum_{i=1}^{n}(X_i - Y_i)^2` 

where *n* - number of non-zero pixels in the mask, *X* - required image (volume), *Y* - comparison image (volume).

MASSIM
""""""

.. _pytorch-msssim: https://github.com/VainF/pytorch-msssim

MASSIM is calculating by the library pytorch-msssim_ 

MASTRESS
""""""""

MASTRESS is calculated by the formula :math:`MASTRESS = \sqrt{1-\frac{(a, b)^2}{||a||_{2}^2||b||_{2}^2}}` 

where *a* - required image (volume), *b* - comparison image (volume), *(a, b)* - the dot product, :math:`||a||_{2}^2` - norm.

MANRMSD
"""""""

MANRMSD is calculated by the formula :math:`MANRMSD = \sqrt{\frac{\sum_{i=1}^{n}(X_i - Y_i)^2}{\sum_{i=1}^{n}X_i^2}}`

where *X* - required image (volume), *Y* - comparison image (volume).

