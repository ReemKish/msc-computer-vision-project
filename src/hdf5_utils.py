#     ┏━━━━━━━━━━━━━━━┓
# ┏━━━┫ hdf5_utils.py ┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃   ┗━━━━━━━━━━━━━━━┛                                              ┃
# ┃ Methods for handling HDF5 files throught the project.            ┃
# ┃ These include:                                                   ┃
# ┃   * Parsing the input HDF5 test set file.                        ┃
# ┃   * Preprocessing the given HDF5 train set.                      ┃
# ┃   * Writing new HDF5 formats to store proccessed train data.     ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
# ====== Imports ====================
# -- python --
import sys
# -- internal --
from const import *
from image_utils import *
# -- external --
import h5py


def find_all(a_str, sub):
    start = 0
    while True:
        start = a_str.find(sub, start)
        if start == -1: return
        yield start
        start += len(sub)

class HDF5_Data():
    def __init__(self, fname):
        self.fname = fname
        self.db = h5py.File(fname, 'r')
        self.datasets = list(self.db['data'].keys())
        self.datasets.remove('word'); self.datasets.remove('font')

    def char_images(self, char: str):
        return self.db['data'][char]['images'][:]

    def char_indices(self, char: str):
        return self.db['data'][char]['indices'][:]

    def char_dataset_size(self, char: str):
        return len(self.db['data'][char]['indices'])

    @property
    def word(self):
        return self.db['data']['word'][:]

    @property
    def font(self):
        return self.db['data']['font'][:]

    def close(self):
        self.db.close()


def convert(infile, outfile):
    """Creates a new HDF5 file from the data in the original file, organized in a different manner."""
    db = h5py.File(infile, 'r')
    f  = h5py.File(outfile, 'w')
    data = f.create_group('data')
    words = [word.decode() for im in db['data'] for word in db['data'][im].attrs['txt']]
    fonts = [FONTS.index(font) for im in db['data'] for font in db['data'][im].attrs['font']]
    word_lengths = [len(word) for word in words]
    characters = ''.join(words)
    n_characters = len(characters)
    word = data.create_dataset('word', n_characters, dtype=np.uint64)
    _    = data.create_dataset('font', n_characters, dtype=np.int64, data=fonts)


    char_images = np.ndarray((*NET_INPUT_SHAPE,n_characters), dtype=np.uint8)
    global_char_idx = 0
    for idx, im in enumerate(db['data']):
        print(f"{im} ({idx}/{len(db['data'])})")
        img = db['data'][im][:]
        charBB = db['data'][im].attrs['charBB']
        for i in range(charBB.shape[-1]):
            char_img = process_bounding_box(img, charBB[:, :, i])
            char_images[:, :, global_char_idx] = char_img
            global_char_idx += 1

    # --- fill 'word' dataset ---
    i = 0; w = 0
    print(word_lengths[:10])
    for l in word_lengths:
        for i in range(i, i+l):
            word[i] = w
        w += 1; i += 1

    # --- fill character datasets ---
    print(characters)
    for char in ''.join(set(characters)):
        group = data.create_group(str(ord(char)))
        indices = list(find_all(characters, char))
        n = len(indices)  # number of occurences of the character in the dataset
        _ = group.create_dataset("indices", n, dtype=np.uint64, data=indices)
        _ = group.create_dataset("images", (*NET_INPUT_SHAPE, n), dtype=np.uint8, data=char_images[:,:,indices])

    f.close()

def main():
    convert("data/imported.h5", "data/converted2.h5")
    db = h5py.File("data/converted2.h5", 'r')
    images = db['data']['97']['images']
    for i in range(images.shape[-1]):
        plt.imshow(images[:,:,i])
        plt.show()

if __name__  == "__main__":
    main()
