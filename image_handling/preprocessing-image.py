# Image preprocessing
# keep in mind, that all this test gonna be implemented
# in the real dataset image

import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
    array_to_img,
    img_to_array,
    load_img
)
import matplotlib.pyplot as plt

# image dir
BASE_DIR = "../datasets/temp/image_train/"
NEW_DIR = ""

def image_handler():

    files = [x for x in os.listdir(BASE_DIR)]

    datage = ImageDataGenerator(rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            brightness_range=[0.4, 1.5])
    
    
    for x in files:

        img = load_img(BASE_DIR+x)
        print(type(img))
        img = img_to_array(img)
        print(img.shape)

        img = img.reshape((1,)+img.shape)
        print(img.shape)
        for batch in datage.flow(img, batch_size=1):
            print(batch)
            print(len(batch))
            img_plot = array_to_img(batch[0])
            plt.figure(1)
            plt.imshow(img_plot)
            print(type(img_plot))
            plt.show()
            break
        break        

def main():
    image_handler()


if __name__ == "__main__":
    main()

