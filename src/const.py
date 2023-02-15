#     ┏━━━━━━━━━━┓
# ┏━━━┫ const.py ┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃   ┗━━━━━━━━━━┛                                                   ┃
# ┃ Various constant values used throughout the project.             ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
from typing import *

SEED = 4

TRAIN_DATASET_FILE = "data/train.h5"

NET_INPUT_SHAPE = (60, 40)
NET_INPUT_SIZE = NET_INPUT_SHAPE[0] * NET_INPUT_SHAPE[1]

FONTS = [
    b'Open Sans',
    b'Sansation',
    b'Titillium Web',
    b'Ubuntu Mono',
    b'Alex Brush',
]

MIN_SAMPLES = 50  # minimum number of samples of a single character required to create a network to recognize it.

# ===== Model Training =====
TRAIN_TEST_SPLIT = .75  # Precent of data used for training, rest is for testing.
# --- Hyperparameters ---
LEARNING_RATE = 2e-3
BATCH_SIZE = 64
CHAR_BATCH_SIZE = 32
EPOCHS = 50
