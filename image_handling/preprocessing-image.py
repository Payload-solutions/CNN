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
BASE_DIR = "../datasets/temp/image_test/"
NEW_DIR = "../model_images/test/"

def image_handler():

    files = [x for x in os.listdir(BASE_DIR)]

    datage = ImageDataGenerator(rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            brightness_range=[0.4, 1.5])
    
    # for this step, we gonna make several changes
    # in the image dataset, first at all in the train set
    # the goal is the implementation of data augmentation
    target_indicator = 529
    for x in files:

        img = load_img(BASE_DIR+x)
        img = img_to_array(img)
        img = img.reshape((1,)+img.shape)
        counter = 0
        
        for batch in datage.flow(img, batch_size=2):
            counter += 1
            img_plot = array_to_img(batch[0])

            img_plot.save(f"{NEW_DIR}test_{target_indicator}.png")
            target_indicator += 1
            if counter == 6:
                break        

def watching_shapes():
    
    list_values = [x for x in os.listdir("../wherever")]

    for x in list_values:

        img = load_img("../wherever/"+x)
        img = img_to_array(img)
        print(img.shape)



def main():
    image_handler()
    # watching_shapes()

if __name__ == "__main__":
    main()

