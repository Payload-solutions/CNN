"""
:author Arturo Negreiros
"""

# utilities for deep model
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, 
    MaxPooling2D, 
    Flatten, 
    Dense, 
    Dropout,
    Activation
)
from tensorflow.keras.datasets import cifar10

# for datacleaning
import numpy as np
import matplotlib.pyplot as plt


def defining_parameters():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()


    # datacleaning
    ## first of all, reduce the dimensionship
    ## of our matrix, come from 0 to 255

    x_train = x_train.astype('float32')/255
    x_test = x_test.astype('float32')/255

    num_classes = len(np.unique(y_train))

    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    # another good practice it's split the set of dataset
    ## trainin, test and validation

    (x_train, x_valid) = x_train[5000:], x_train[:5000]
    (y_train, y_valid) = y_train[5000:], y_train[:5000]



def main():
    defining_parameters()



if __name__ == "__main__":
    main()