# Intro to Computer Vision - Final Project
Re'em Kishinevsky 213057094

Download all the code and puts all .py source files in the same directory.
Download the models zipfile from Dropbox (https://www.dropbox.com/s/14k7gi6nu68dwnz/best_models.zip?dl=0)
and unzip it in this directory.
Make sure accuracies.json and all the model .pth files are directly in the folder named `best_models`.
The directory tree should now look like this:

.
├── best_models
│   ├── accuracies.json
│   ├── char_a.pth
│   ├── char_b.pth
│   ├──   .. 
│   ├──   .. 
│   ├──   ..
│   └── global.pth
├── const.py
├── dataset.py
├── hdf5_utils.py
├── image_utils.py
├── models.py
├── plot.py
├── predict.py
├── preprocess.py
├── Readme.md
├── training.py
├── train.py
└── types_.py

(Make sure to open the file as text for the above illustration to make sense)

Then, run:

```
  python3 predict.py hdf5file
```

where hdf5file contains the test data.
You may be required to install some dependency packages, but they are
most likely already installed: e.g: torchvision, cv2, matplotlib.

The results will be written into results.csv.

The code is also available on GitHub at (https://github.com/ReemKish/msc-computer-vision-project)

~ Re'em
