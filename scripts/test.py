import numpy as np
import cv2

def create_square_mask(radius: int) -> np.ndarray:
    mask = cv2.rectangle(np.zeros((256, 256)), (128-radius, 128-radius), 
                         (127+radius, 127+radius), (1), -1)
    return mask

def create_crop(ref: np.ndarray, radius: int) -> np.ndarray:
    img = ref.copy()
    return img[(128-radius):(128+radius), (128-radius):(128+radius)]

def create_reference(radius: int):
    img_1 = np.zeros((256, 256))
    img_2 = np.zeros((256, 256))
    img_2[128, 128] = 1
    mask = create_square_mask(radius)
    result = 1/((2*radius)**2)
    return img_1, img_2, mask, result