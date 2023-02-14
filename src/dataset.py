#     ┏━━━━━━━━━━━━┓
# ┏━━━┫ dataset.py ┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃   ┗━━━━━━━━━━━━┛                                                 ┃
# ┃ Train and test datasets derived from the given HDF5 files.       ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
# ====== Imports ====================
# -- python standard library --
from collections import namedtuple
# -- internal --
from const import *
from types_ import *
from hdf5_utils import *
# -- external --
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset, random_split
from torchvision.transforms import ToTensor, Lambda
import cv2 as cv

class CharacterDataset(Dataset):
    def __init__(self, character: str, images: ArrayNxMx3xK, font : ArrayN, word : ArrayN):
        self.character = character
        self.images = images
        self.font = font
        self.word = word
        self.to_tensor = ToTensor()

    def __len__(self):
        return self.images.shape[-1]

    def transform(self, img: ArrayNxM[np.uint8]):
        # gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray_img = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
        return gray_img
        # canny = cv.Canny(gray_img, 50, 100, apertureSize=3)
        # return img
        # return cv.resize(img, (50,50))
        # gray = cv.cvtColor(cv.cvtColor(img, cv.COLOR_BGR2GRAY), cv.COLOR_GRAY2BGR)
        # canny_gray = cv.cvtColor(cv.Canny(gray, 50, 100, apertureSize=3), cv.COLOR_GRAY2BGR)
        # canny_rgb  = cv.cvtColor(cv.Canny(img, 50, 100, apertureSize=3), cv.COLOR_GRAY2BGR)
        # print(f"{img.shape=}, {gray.shape=}, {canny_gray.shape=}, {canny_rgb.shape=}")
        # out = np.zeros_like(img)
        # out = img*0.5 + canny
        # row1 = np.concatenate((gray, img))
        # row2 = np.concatenate((canny_gray, canny_rgb))
        # all = np.concatenate((row1, row2), axis=1)
        # plt.imshow(all)
        # plt.show()
        # return img
    

    def __getitem__(self, idx):
        image = self.images[:, :, idx]
        label = self.font[idx]
        target_transform = Lambda(lambda y: torch.zeros(5, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))
        image = (self.transform(image))
        # image = image.swapaxes(0, 2).swapaxes(0,1)
        image = self.to_tensor(image)
        label = target_transform(label)
        return image, label


def create_dataloaders(fname: str) -> Dict[str, TrainTestData]:
    """Creates dataloaders for each character from the data given in the preprocessed HDF5 file `fname`.

    Returns a tuple (character_dl, global_dl):
    character_dls:
        dictionary mapping each character to a namedtuple of (train_dataloader, test_dataloader)
        derived from the corresponding CharacterDataset.
    global_dl:
        namedtuple of (train_dataloader, test_dataloader) drived from the concatenation of all characer datasets.
    """
    datafile = HDF5_Data(fname)
    character_dls = dict()
    all_char_datasets = []
    for char in datafile.datasets:
        if datafile.char_dataset_size(char) < MIN_SAMPLES: continue
        font = datafile.font[datafile.char_indices(char)]
        word = datafile.word[datafile.char_indices(char)]
        images = datafile.char_images(char)
        char_ds = CharacterDataset(char, images, font, word)
        all_char_datasets.append(char_ds)
        train_data, test_data = random_split(char_ds, [TRAIN_TEST_SPLIT, 1 - TRAIN_TEST_SPLIT])
        train_dataloader = DataLoader(train_data, batch_size=CHAR_BATCH_SIZE, shuffle=True, pin_memory=True)
        test_dataloader = DataLoader(test_data, batch_size=CHAR_BATCH_SIZE, shuffle=True, pin_memory=True)
        character_dls[chr(int(char))] = TrainTestData(train_dataloader, test_dataloader)
    datafile.close()
    global_ds = ConcatDataset(all_char_datasets) 
    global_train_data, global_test_data = random_split(global_ds, [TRAIN_TEST_SPLIT, 1 - TRAIN_TEST_SPLIT])
    global_train_dl = DataLoader(global_train_data, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    global_test_dl = DataLoader(global_test_data, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    global_dl = TrainTestData(global_train_dl, global_test_dl)
    return character_dls, global_dl

def main():
    character_dls, global_dl = create_dataloaders('data/converted.h5')
    dl = character_dls['e'].train_dataloader
    for im, label in dl:
        plt.imshow(im)
        print(label)
        plt.show()

if __name__ == "__main__":
    main()
