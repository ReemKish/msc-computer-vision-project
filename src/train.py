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
from plot import dataset_first_n, plot_fit
# -- external --
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

character_data, global_data = create_dataloaders("data/augmented.h5")
# dl_train, dl_test = character_dls['L']

def show():
    # while True:
    # dataset_first_n(dl_train.dataset, 64, False, nrows=4, cmap='gray')
    # plt.show()
    pass


def train_model(name, data : TrainTestData, epochs=EPOCHS, model=None):
    """Trains the global model. saving the model at it's best-accuracy version to a file."""
    if not model:
        model = CNN((1, *NET_INPUT_SHAPE), 5)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = Trainer(model, loss_fn, optimizer, device, name)
    dl_train, dl_test = data
    return trainer.fit(dl_train, dl_test, epochs, early_stopping=3)

def train_models(train_global=True):
    if train_global:  # trains global model
        global_model = torch.load("./best_models/global.pth")
        fit_result = train_model("global", global_data, model=global_model)
        plot_fit(fit_result, legend=True)
        plt.show()
    for character in character_data:
        global_model = torch.load("./best_models/global.pth")
        try:
            data = character_data[character]
            train_model(f"char_{character}", data, epochs=EPOCHS//2, model=global_model)
        except Exception:
            continue


if __name__ == "__main__":
    train_models(train_global=True)

