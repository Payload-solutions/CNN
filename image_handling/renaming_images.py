"""
This source code is for renaming the data images
to make the classification categorical, will be 
more easy to catch for the neural architecture
"""

import re
import os
import sys
import pandas as pd
from tensorflow.keras.preprocessing.image import (
    array_to_img,
    load_img,
    img_to_array
)


def renaiming_images(file_path: str, name: str) -> None:

    dataset = pd.read_csv("../model_images/milk-properties.csv")
    new_path = f"../image_set/{name}/"

    try:
        files = os.listdir(file_path)
        for pos, x in enumerate(files):
            value = re.findall(r"\d{1,5}", x)
            data_splited = dataset.iloc[int(value[0])]["quality_product"].split()
            new_name = "_".join(data_splited)
            img = load_img(file_path+x)

            img.save(f"{new_path}{new_name}.{pos}.png")
        print("Finished..")
    except Exception as e:
        print(str(e))
    except KeyboardInterrupt:
        exit(0)

def main():
    
    train_path = "../model_images/train/"
    test_path = "../model_images/test/"
    valid_path = "../model_images/validator/"

    # renaiming_images(train_path, "train")
    renaiming_images(test_path, "test")
    renaiming_images(valid_path, "validator")




if __name__ == "__main__":
    main()
