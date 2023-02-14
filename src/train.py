#     ┏━━━━━━━━━━┓
# ┏━━━┫ train.py ┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃   ┗━━━━━━━━━━┛                                                   ┃
# ┃ Train.                                                           ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
# ====== Imports ====================
# -- internal --
from const import *
from types_ import *
from dataset import create_dataloaders
from models import DNN, CNN, FCN
from training import Trainer
from plot import dataset_first_n
# -- external --
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

character_dls, global_dl  = create_dataloaders("data/augmented2.h5")
# dl_train, dl_test = character_dls['L']
dl_train, dl_test = global_dl

def show():
    # while True:
    # dataset_first_n(dl_train.dataset, 64, False, nrows=4, cmap='gray')
    # plt.show()
    pass


def train_model():
    """Trains the global model. saving the model at it's best-accuracy version to a file."""
    model = CNN((1, *NET_INPUT_SHAPE), 5)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = Trainer(model, loss_fn, optimizer, device, "global")
    trainer.fit(dl_train, dl_test, EPOCHS, early_stopping=3)



def train_models(train_global=True):
    if train_global:  # trains global model
        pass


if __name__ == "__main__":
    train_models(train_global=False)
