import numpy as np
from metrics import *
import pytest

def create_reference(radius: int):
    img_1 = np.zeros((256, 256))
    img_2 = np.zeros((256, 256))
    img_2[128, 128] = 1
    mask = create_square_mask(radius)
    result = 1/((2*radius)**2)
    return img_1, img_2, mask, result

def test_():
    rad = 5
    img_1, img_2, mask, res = create_reference(rad)
    crop_mask = create_crop(mask, rad)
    assert np.count_nonzero(crop_mask) == (rad*2)**2
    val = MAMSE(img_1, img_2, np.ones((256, 256)))
    print(val)
    with pytest.raises(ImError): 
        MAMSE(img_1, img_2, np.ones((255, 256)))

test_()