import metrics
import numpy as np
import matplotlib.pyplot as plt
import pytest

def create_reference(radius: int):
    img_1 = np.ones((256, 256))
    img_2 = np.ones((256, 256))
    img_2[128, 128] = 0
    mask = metrics.create_square_mask(radius)
    result = 1/((2*radius)**2)
    return img_1, img_2, mask, result

def stress_ref():
    img_1 = np.ones((2, 256, 256))
    img_2 = np.ones((2, 256, 256))
    img_2[:, 128, 128] = 0
    img_2[1, 128, 127] = 0
    return img_1, img_2

def test_():
    '''rad = 5
    img_1, img_2, mask, res = create_reference(rad)
    crop_mask = metrics.create_crop(mask, rad)
    assert np.count_nonzero(crop_mask) == (rad*2)**2

    mse = metrics.MAMSE(img_1, img_2, mask)
    assert mse == res'''

    img_3, img_4 = stress_ref()
    print((metrics.MANRMSD(img_3, img_4, per_slice_avg=True)), metrics.MANRMSD(img_3, img_4))
test_()