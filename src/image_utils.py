#     ┏━━━━━━━━━━━━━━━━┓
# ┏━━━┫ image_utils.py ┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃   ┗━━━━━━━━━━━━━━━━┛                                             ┃
# ┃ Various image manipulation and processing utility functions.     ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

# ====== Imports ====================
# -- internal --
from const import *
from types_ import *
# -- external --
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

def minimal_containing_rectangle(img : ArrayNxMx3[np.uint8], polygon : ArrayNx2[np.float32]):
    """
    Returns the minimal axis-aligned cropped rectangle of the image that contains the
    polygon with all pixels outside of it blacked.
    """

    height, width, _ = img.shape
    max_x  = int(np.min((np.ceil (np.max(polygon[:, 0])), width)))
    min_x  = int(np.max((np.floor(np.min(polygon[:, 0])), 0)))
    max_y  = int(np.min((np.ceil (np.max(polygon[:, 1])), height)))
    min_y  = int(np.max((np.ceil (np.min(polygon[:, 1])), 0)))
    cropped = img[min_y:max_y,min_x:max_x,:]

    plt.imshow(cropped)
    plt.show()
    

def process_bounding_box(img : ArrayNxMx3[np.uint8], polygon : ArrayNx2[np.float32]):
    """Returns a sterilized small axis-aligned gray-scale image of the bounded character."""
    polygon = np.round(polygon).astype(int).transpose()
    rect = cv.minAreaRect(polygon)
    box = cv.boxPoints(rect)
    box = np.int0(box)
    # based on post at https://goo.gl/Q92hdp
    center = rect[0]
    size = rect[1]
    angle = rect[2]
    center, size = tuple(map(int, center)), tuple(map(int, size))
    rows, cols = img.shape[0], img.shape[1]
    M = cv.getRotationMatrix2D(center, angle, 1)
    img_rot = cv.warpAffine(img, M, (cols, rows))
    out = cv.getRectSubPix(img_rot, size, center)
    if out.shape[0] < out.shape[1]:
        out = cv.rotate(out, cv.ROTATE_90_CLOCKWISE)
    out = cv.resize(out, NET_INPUT_SHAPE)
    out = cv.cvtColor(out, cv.COLOR_BGR2GRAY)
    return out



def normalize_bb():
    pass


