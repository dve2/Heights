# Introduction
The project is dedicated to the computation of globular object heights in scanning probe microscopy (SPM) images using a convolutional neural network with U-net architecture (model 1). The input of the model is a two-channel image. The first channel is the analysed image and the second channel is a binary mask containing only the highest pixels of objects in the image. The loss function and metric of the model is the mean square error (MSE) between the model prediction and a pre-plotted map of object heights (target). The target has the same dimensionality as the input image and contains non-zero pixels, height values, only at the positions of the highest points of objects in the image, similar to the binary mask. The MSE is computed for the non-zero pixels of the binary mask/target. 
The model was trained on 187 images and validated and tested on 38 and 18 images respectively. The model produces a prediction in the form of a single-channel height map with the same dimensionality as the input image. The binary mask from input channel 2 is then applied to the model prediction, leaving only the pixels at the positions of the highest pixel of each object. The training resulted in a metric on the test dataset of ~0.29 nm2.

To predict the heights of the globular objects on the user data (inference), the second convolutional neural network with U-net architecture (model 2) was trained, which finds the objects in an image and creates the binary mask of the highest pixels of objects in an image for the model 1 input.

The inference ("predict.py" file) works as follows:
The SPM image file is required for input. The SPM image file should be in txt format. The architecture of the format can be found in the example files in the Images folder. The SPM images obtained on the different microscopes were converted to a txt format using the 'Export' button in FemtoScan software.

The output of the inference contains 3 parts: 
1. The source image with the enumerated objects
2. The txt file with the height of each object. The order of the height values in the txt file corresponds to the enumeration in the image.
3. The histogram of the height distribution.


# Installation

For local run.

Clone the repo:

    git clone https://github.com/dve2/Heights.git

Make [virtual environment](https://docs.python.org/3/library/venv.html):

    python -m venv .venv
    

Activate .venv and install required packages:

    source venv/bin/activate
    pip install -r requirements.txt
    

For use in colab:

    TODO add colab notebook for inference


# Inference

1. Open project folder in terminal

        cd Heights

2. Copy **input files** (TODO : change to proper names of microscope outputs)
to [Inference folder](Inference)
3. Activate virtual environment

       source venv/bin/activate
4. Run [predict.py](predict.py)

       python predict.py
5. Results will be saved into 




Example of working inference (one file only)
[inference example](https://colab.research.google.com/github/dve2/Heights/blob/main/notebooks/inference_new.ipynb)

Example of working inference (any file from test dataset; downloads big file):
[inference example](https://colab.research.google.com/github/dve2/Heights/blob/main/notebooks/inference.ipynb)


# Training

[train example](https://colab.research.google.com/github/dve2/Heights/blob/main/notebooks/Train_2ch_ml_dm.ipynb)

Training dataset available by request

