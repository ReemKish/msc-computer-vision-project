#     ┏━━━━━━━━━━┓
# ┏━━━┫ model.py ┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃   ┗━━━━━━━━━━┛                                                   ┃
# ┃ Our ML model used for font detection.                            ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
# ====== Imports ====================
# -- internal --
from const import *
from types_ import *
from dataset import create_dataloaders
# -- external --
import matplotlib.pyplot as plt
import torch
from torch import nn


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


class FCN(nn.Module):
    """
    A Fully Convolutional Network (FCN).
    Can accept dynamically-shaped input.
    """
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
           
        )


class DNN(nn.Module):
    """
    A simple fully-connected classifier model.
    """
    def __init__(self):
        super().__init__()
        inpt_size = NET_INPUT_SHAPE[0] * NET_INPUT_SHAPE[1] * NET_INPUT_SHAPE[2]
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(inpt_size, 1024),
            nn.PReLU(),
            nn.Linear(1024, 512),
            nn.Dropout(0.2),
            nn.PReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 1024),
            nn.Dropout(0.2),
            nn.PReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.PReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.BatchNorm1d(256),
            nn.Linear(256, 64),
            nn.PReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 5),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.flatten(x)
        out = self.layers(x)
        return out

class CNN(nn.Module):
    """
    A convolutional classifier model.

    The architecture is:
    [(Conv -> ReLU)*P -> MaxPool]*(N/P) -> (Linear -> ReLU)*M -> Linear
    """
    def __init__(self, in_size, out_classes, filters, pool_every, hidden_dims):
        """
        :param in_size: Size of input images, e.g. (C,H,W).
        :param out_classes: Number of classes to output in the final layer.
        :param filters: A list of of length N containing the number of
            filters in each conv layer.
        :param pool_every: P, the number of conv layers before each max-pool.
        :param hidden_dims: List of of length M containing hidden dimensions of
            each Linear layer (not including the output layer).
        """
        super().__init__()
        self.in_size = in_size
        self.out_classes = out_classes
        self.filters = filters
        self.pool_every = pool_every
        self.hidden_dims = hidden_dims

        self.feature_extractor = self._make_feature_extractor()
        self.classifier = self._make_classifier()

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the feature extractor part of the model:
        # [(Conv -> ReLU)*P -> MaxPool]*(N/P)
        # Use only dimension-preserving 3x3 convolutions. Apply 2x2 Max
        # Pooling to reduce dimensions.
        # ====== YOUR CODE: ======
        layers = [
            nn.Conv2d(in_channels, 64, 3),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 32, 3),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.MaxPool2d(2),
        ]
        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def _make_classifier(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the classifier part of the model:
        # (Linear -> ReLU)*M -> Linear
        # You'll need to calculate the number of features first.
        # The last Linear layer should have an output dimension of out_classes.
        # ====== YOUR CODE: ======
        def num_flat_features(x):
            size = x.size()[1:]  # all dimensions except the batch dimension
            num_features = 1
            for s in size:
                num_features *= s
            return num_features

        M = len(self.hidden_dims)
        n_channels = num_flat_features(self.feature_extractor(torch.empty(1, in_channels, in_h, in_w))[1:])
        # layers += [ nn.Linear(n_channels, self.hidden_dims[0]), nn.ReLU() ]  # first fc
        # for i in range(1, M):
        #     layers += [
        #         nn.Linear(self.hidden_dims[i-1], self.hidden_dims[i]),
        #         nn.ReLU()
        #     ]
        # layers += [ nn.Linear(self.hidden_dims[-1], self.out_classes) ]  # last fc
        # layers += [ nn.Softmax(dim=1) ]
        # # ========================
        # seq = nn.Sequential(*layers)
        seq = nn.Sequential(
            nn.Linear(n_channels, 1024),
            nn.PReLU(),
            nn.Linear(1024, 512),
            nn.Dropout(0.5),
            nn.PReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 1024),
            nn.Dropout(0.5),
            nn.PReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.PReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.PReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(256),
            nn.Linear(256, 64),
            nn.PReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 5),
            nn.LogSoftmax(dim=1)
        )
        return seq

    def forward(self, x):
        # TODO: Implement the forward pass.
        # Extract features from the input, run the classifier on them and
        # return class scores.
        # ====== YOUR CODE: ======
        def num_flat_features(x):
            size = x.size()[1:]  # all dimensions except the batch dimension
            num_features = 1
            for s in size:
                num_features *= s
            return num_features

        features = self.feature_extractor(x)
        features = features.view(-1, num_flat_features(features))
        out = self.classifier(features)
        # ========================
        return out

def main():
    pass

if __name__ == "__main__":
    main()
