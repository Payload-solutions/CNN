# Image preprocessing
# keep in mind, that all this test gonna be implemented
# in the real dataset image

import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# image dir
BASE_DIR = "../datasets/temp/image_train/"

def image_handler():

    files = [x for x in os.listdir(BASE_DIR)]
    print(files)



def main():
    image_handler()


if __name__ == "__main__":
    main()
