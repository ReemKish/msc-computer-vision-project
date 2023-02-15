#     ┏━━━━━━━━━━━━┓
# ┏━━━┫ predict.py ┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃   ┗━━━━━━━━━━━━┛                                                 ┃
# ┃ Predicts fonts in an image.                                      ┃
# ┃ Receives an HDF5 file from the commandline and outputs a CSV     ┃
# ┃ file with the results.                                           ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
# ====== Imports ====================
# -- python --
import sys
# -- python --
from math import ceil
# -- internal --
from const import *
from image_utils import *
from hdf5_utils import *
# -- external --
import pickle
import csv
import torch
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import json
import os


device = "cuda" if torch.cuda.is_available() else "cpu"
tensorize = ToTensor()
global_model = None
character_models = None
accuracies = None
data = None

def classify(words: List[List[Character]], transform = lambda a: a) -> Dict[int, Tuple[str, int]]:
    """Classifies the characters given the various models' predictions as Character objects grouped into words.

    Returns a mapping `fonts` from a character's index to it's character and predicted font index.
    :param transform: function that maps a model's accuracy (0.01-0.99) to a scalar multipled
                      by its prediction vector and thus decides the prediction's influence
                      on the final classification.
    """
    fonts = dict()
    for word in words:
        acc = lambda character: accuracies[f'char_{character.char}'] if f'char_{character.char}' in accuracies else accuracies['global']
        pred = sum(character.pred * transform(acc(character)) for character in word)
        font = pred.argmax()
        for character in word:
            fonts[character.idx] = (character.char, font)
    return fonts

def convert_results_format(predictions):
    """Convert the prediction results format so results are grouped by words."""
    results = [[] for _ in range(int(np.max(data.word))+1)]
    word = data.word
    for char in predictions:
        indices = data.char_indices(str(ord(char)))
        char_predictions = predictions[char]
        for i, idx in enumerate(indices):
            results[word[idx]].append(Character(idx, char, np.exp(char_predictions[i])))
    return results

def load_models():
    """Loads the models saved in folder './best_models'

    Returns a tuple (global_model, character_models, accuracies) where:
        global_model - the global model trained to classify all characters.
        character_models - a dictionary mapping each character to the model that
            classifies fonts for this specific character.
        accuracies - a dictionary mapping a model's name to its accuracy score.
    """
    global global_model, character_models, accuracies
    global_model = None
    character_models = dict()
    accuracies = dict()
    # read accuracies json file:
    with open('./best_models/accuracies.json', 'r') as f:
        accuracies = json.load(f)
    # filter all models with accuracies less than the global model
    accuracies = {model : acc for model, acc in accuracies.items() if acc >= accuracies['global']}

    # load individual characters models
    for model_file in [f for f in os.listdir('./best_models') if f.startswith('char_') and f[:-4] in accuracies]:
        char = model_file[len('char_')]
        model = torch.load(f'./best_models/{model_file}')
        model.eval()
        character_models[char] = model
    global_model = torch.load('./best_models/global.pth')
    global_model.eval()

    return global_model, character_models, accuracies


def calculate_char_predictions(batch_size=BATCH_SIZE):
    """Returns a dictionary mapping every character to an array of predictions for each of its occurences in the data."""
    predictions = dict()
    for char_ord in data.datasets:
        char = chr(int(char_ord))
        if f"char_{char}" in accuracies:  # exist a specific model for this char with better accuracy than the global model.
            model = character_models[char]
        else: model = global_model
        images = data.char_images(str(ord(char)))
        for i in range(images.shape[0]):
            images[i] = cv.normalize(images[i], None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
        num_batches = ceil(images.shape[0]/batch_size)
        batches = [images[batch_size*y:batch_size*(y+1),:,:] for y in range(num_batches)]
        pred = np.ndarray((images.shape[0], 5), dtype=np.float32)
        for i, batch in enumerate(batches):
            # tmp = cv.normalize(tmp, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
            # print(tmp)
            # batch = cv.normalize(batch, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
            # print(batch[0])
            # print("-------------")
            batch = tensorize(batch).swapaxes(0,1).swapaxes(1,2).to(device)
            with torch.no_grad():
                batch_pred = model(batch.unsqueeze(1))
            pred[i*batch_size:(i+1)*batch_size,:] = batch_pred.cpu()
        predictions[char] = pred
    return predictions

def write_results_to_csv(fonts: Dict[int,int], im_names, csv_fname='results.csv'):
    with open(csv_fname, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['','image','char','Open Sans','Sansation','Titillium Web','Ubuntu Mono','Alex Brush'])
        for idx in sorted(fonts.keys()):
            onehot_font_vec = [0, 0, 0, 0, 0]
            onehot_font_vec[fonts[idx][1]] = 1
            writer.writerow([idx, im_names[idx], fonts[idx][0], *onehot_font_vec])


def main():
    global data
    if len(sys.argv) < 2:
        print("Error: missing argument <h5file>")
    else:
        hd5file = sys.argv[1]
        im_names = convert(hd5file, "ProcessedTestData.h5", labels=False)
        with open('im_names_test.pkl', 'wb') as f: pickle.dump(im_names, f)
        with open('im_names_test.pkl', 'rb') as f: im_names = pickle.load(f)
        data = HDF5_Data("ProcessedTestData.h5")
        load_models()
        predictions = calculate_char_predictions()
        words = convert_results_format(predictions)
        fonts = classify(words)
        write_results_to_csv(fonts, im_names)
    

if __name__ == "__main__":
    main()
