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
    width, height = rect[1]
    size1 = (width, height)
    size2 = (width + 5, height + 5)
    angle = rect[2]
    center = tuple(map(int, center))
    # size1   = tuple(map(int, size1))
    size2   = tuple(map(int, size2))
    rows, cols = img.shape[0], img.shape[1]
    M = cv.getRotationMatrix2D(center, angle, 1)
    img_rot = cv.warpAffine(img, M, (cols, rows))
    # out1 = cv.getRectSubPix(img_rot, size1, center)
    out2 = cv.getRectSubPix(img_rot, size2, center)
    # if out1.shape[0] < out1.shape[1]: out1 = cv.rotate(out1, cv.ROTATE_90_CLOCKWISE)
    if out2.shape[0] < out2.shape[1]: out2 = cv.rotate(out2, cv.ROTATE_90_CLOCKWISE)
    # out1 = cv.resize(out1, (NET_INPUT_SHAPE[1], NET_INPUT_SHAPE[0]))
    out2 = cv.resize(out2, (NET_INPUT_SHAPE[1], NET_INPUT_SHAPE[0]))
    out2 = cv.cvtColor(out2, cv.COLOR_BGR2GRAY)

    # plt.imshow(np.concatenate((out1, out2), axis=1))
    # plt.show()
    return out2



def normalize_bb():
    pass


