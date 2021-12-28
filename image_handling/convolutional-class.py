
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (
    Dense,
    Conv2D,
    Activation,
    Dropout,
    MaxPooling2D,
    Flatten
)
from tensorflow.keras.models import Sequential
import pandas as pd





class Dataset:

    def __init__(self, base_dir: str):
        """
        Args:
            base_dir (str): description for the current location of images
        """
        self.base_dir = base_dir
        self.model = Sequential()
        self.regularizers = 1e-4
    
    def creating_model():

        self.model.add(Conv2D(filters, kernel_size))

class ConvModel:
    def __init__(self):
        pass