# First test case for the convolutional neural networks
# using tensorflow
# @author: Arturo Negreiros Pxyl0xd


import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import (
    Conv2D,
    Dropout,
    MaxPooling2D,
    Flatten,
    Dense
)
import numpy as np
import matplotlib.pyplot as plt


def main():
    (train_image, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    
    # normalizing images
    train_image = train_image.astype('float32')/255
    test_images = test_images.astype('float32')/255

    # reshaping the images
    train_image = train_image.reshape(train_image.shape[0], 28,28,1)
    test_images = test_images.reshape(test_images.shape[0], 28,28,1)

if __name__ == "__main__":
    main()
