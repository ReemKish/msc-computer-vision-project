#     ┏━━━━━━━━━━━━━━━┓
# ┏━━━┫ preprocess.py ┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃   ┗━━━━━━━━━━━━━━━┛                                              ┃
# ┃ Preprocesses an HDF5 file structured according to the project    ┃
# ┃ specification. Results in a new HDF5 with the preprocessed data. ┃
# ┃ Preprocessing include:                                           ┃
# ┃   * Restructring of the HDF5 file such that the data is grouped  ┃
# ┃     by character rather than by image.                           ┃
# ┃   * Cutting patches of characters from an image according to     ┃
# ┃     their bounding boxes, then rotating them to be axis-aligned. ┃
# ┃   TODO : applying Canny each detector.                           ┃
# ┃                                                                  ┃
# ┃                                                                  ┃
# ┃                                                                  ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

from const import *
from image_utils import *
import matplotlib.pyplot as plt
import h5py
from sys import argv

def get_char_count(db):
    char_count = dict()
    for im_name in db['data']:
        for word in db['data'][im_name].attrs['txt']:
            for char in word:
                if char in char_count: char_count[char] += 1
                else: char_count[char] = 1
    return char_count

def get_all_characters(db):
    all_characters = bytearray()
    for im_name in db['data']:
        for word in db['data'][im_name].attrs['txt']:
            all_characters += word
    return all_characters


def convert(fname):
    """Creates a new HDF5 file from the data in the original file, organized in a different manner."""
    db = h5py.File(fname, 'r')
    char_count = get_char_count(db)
    all_characters = get_all_characters(db)
    n_chars = len(all_characters)
    f = h5py.File(TRAIN_DATASET_FILE, 'w')
    data = f.create_group("data")
    attr_font_dict = dict()
    attr_word_dict = dict()
    for char in [c for c in char_count if char_count[c] > 0]:
        data.create_dataset(f'{char}', (*NET_INPUT_SHAPE,char_count[char]*2), dtype=(np.uint8))
        attr_font_dict[char] = np.zeros((char_count[char]*2,), dtype='int64')
        attr_word_dict[char] = np.zeros((char_count[char]*2,), dtype='int64')
    global_char_ind = 0
    global_word_ind = 0
    cur_word_ind = 0
    cur_char_indices = {char : 0 for char in all_characters}
    for i, im_name in enumerate(db['data'].keys()):
        print(f"{im_name} ({i}/{len(db['data'])})")
        image_char_ind = 0
        image_word_ind = 0
        cur_word_ind = 0
        img = db['data'][im_name][:]
        charBB = db['data'][im_name].attrs['charBB']
        n_image_chars = charBB.shape[-1]
        for j in range(n_image_chars):
            if global_char_ind >= n_chars:
                for char in char_count:
                    data[f'{char}'].attrs.create('font', attr_font_dict[char])
                    data[f'{char}'].attrs.create('word', attr_word_dict[char])
                f.close()
                db.close()
                return

            char = all_characters[global_char_ind]
            char_img = process_bounding_box(img, charBB[:, :, j])
            data[f'{char}'][:, :, cur_char_indices[char]+0] = char_img
            data[f'{char}'][:, :, cur_char_indices[char]+1] = cv.rotate(char_img, cv.ROTATE_180) 
            # data[f'{char}'][:, :, cur_char_indices[char]+2] = cv.rotate(char_img, cv.ROTATE_90_CLOCKWISE)
            # data[f'{char}'][:, :, cur_char_indices[char]+3] = cv.rotate(char_img, cv.ROTATE_90_COUNTERCLOCKWISE) 
            font = FONTS.index(db['data'][im_name].attrs['font'][image_char_ind])
            attr_font_dict[char][cur_char_indices[char]+0] = font
            attr_font_dict[char][cur_char_indices[char]+1] = font
            # attr_font_dict[char][cur_char_indices[char]+2] = f
            # attr_font_dict[char][cur_char_indices[char]+3] = f
            attr_word_dict[char][cur_char_indices[char]+0] = global_word_ind
            attr_word_dict[char][cur_char_indices[char]+1] = global_word_ind
            # attr_word_dict[char][cur_char_indices[char]+2] = global_word_ind
            # attr_word_dict[char][cur_char_indices[char]+3] = global_word_ind

            if cur_word_ind >= len(db['data'][im_name].attrs['txt'][image_word_ind]) - 1:
                global_word_ind += 1
                image_word_ind += 1
                cur_word_ind = -1

            # increment indices
            global_char_ind += 1
            image_char_ind += 1
            cur_char_indices[char] += 2
            cur_word_ind += 1
            # plt.imshow(char_img)
            # plt.show()

    for char in char_count:
        data[f'{char}'].attrs.create('font', attr_font_dict[char])
        data[f'{char}'].attrs.create('word', attr_word_dict[char])
    f.close()
    db.close()






def main():
    fname = argv[1]
    convert(fname)
    db = h5py.File(TRAIN_DATASET_FILE, 'r')
    # print(db['data'][f'{ord("m")}'].attrs['word'])
    # print(db['data'][f'{ord("m")}'].attrs['font'])
    

if __name__ == "__main__":
    main()
