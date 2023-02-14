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
dl_train, dl_test = character_dls['a']
# dl_train, dl_test = global_dl

def show():
    # while True:
    # dataset_first_n(dl_train.dataset, 64, False, nrows=4, cmap='gray')
    # plt.show()
    pass

def main():
    model = CNN((1, *NET_INPUT_SHAPE), 5, filters=[16, 32, 64], pool_every=5, hidden_dims=[64, 128, 256])
    # model = DNN()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = Trainer(model, loss_fn, optimizer, device)
    trainer.fit(dl_train, dl_test, EPOCHS, early_stopping=3)


if __name__ == "__main__":
    show()
    main()
