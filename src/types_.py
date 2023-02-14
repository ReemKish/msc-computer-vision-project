#     ┏━━━━━━━━━━┓
# ┏━━━┫ types.py ┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃   ┗━━━━━━━━━━┛                                                   ┃
# ┃ Various types used throughout the project.                       ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
# ====== Imports ====================
# -- python standard library --
from collections import namedtuple
from typing import *
# -- external --
import numpy.typing as npt
import numpy as np

DType = TypeVar("DType", bound=np.generic)

ArrayNxMx3 = Annotated[npt.NDArray[DType], Literal['N', 'M', 3]]
ArrayNxM = Annotated[npt.NDArray[DType], Literal['N', 'M']]
ArrayNxMxK = Annotated[npt.NDArray[DType], Literal['N', 'M', 'K']]
ArrayKxNxM = Annotated[npt.NDArray[DType], Literal['K', 'N', 'M']]
ArrayKx4xNxM = Annotated[npt.NDArray[DType], Literal['K', 4, 'N', 'M']]
ArrayKxNxMx5 = Annotated[npt.NDArray[DType], Literal['K', 'N', 'M', '5']]
ArrayNxMx3xK = Annotated[npt.NDArray[DType], Literal['N', 'M', 3, 'K']]
ArrayNx2 = Annotated[npt.NDArray[DType], Literal['N', 2]]
ArrayKx5 = Annotated[npt.NDArray[DType], Literal['K', 5]]
ArrayN = Annotated[npt.NDArray[DType], Literal['N']]
Array5xN = Annotated[npt.NDArray[DType], Literal[5, 'N']]
Array5 = Annotated[npt.NDArray[DType], Literal[5]]
TrainTestData = namedtuple("TrainTestData", ('train_dataloader', 'test_dataloader'))

class Character(NamedTuple):
    """
    Represents a single character.
    """
    idx : int
    char : str
    pred: Array5[np.float]




class FitResult(NamedTuple):
    """
    Represents the result of fitting a model for multiple epochs given a
    training and test (or validation) set.
    The losses are for each batch and the accuracies are per epoch.
    """
    num_epochs: int
    train_loss: List[float]
    train_acc: List[float]
    test_loss: List[float]
    test_acc: List[float]

class BatchResult(NamedTuple):
    """
    Represents the result of training for a single batch: the loss
    and number of correct classifications.
    """
    loss: float
    num_correct: int


class EpochResult(NamedTuple):
    """
    Represents the result of training for a single epoch: the loss per batch
    and accuracy on the dataset (train or test).
    """
    losses: List[float]
    accuracy: float


class FitResult(NamedTuple):
    """
    Represents the result of fitting a model for multiple epochs given a
    training and test (or validation) set.
    The losses are for each batch and the accuracies are per epoch.
    """
    num_epochs: int
    train_loss: List[float]
    train_acc: List[float]
    test_loss: List[float]
    test_acc: List[float]

