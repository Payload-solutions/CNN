"""
This source code is for renaming the data images
to make the classification categorical, will be 
more easy to catch for the neural architecture
"""

import re
import os
import sys
import pandas as pd


def renaiming_images(file_path: str, name: str) -> None:

    dataset = pd.read_csv("../model_images/milk-properties.csv")

    files = os.listdir(file_path)
    for x in files:
        value = re.findall(r"\d{1,5}", x)
        new_name = "_".join(dataset.iloc[int(value[0])]["quality_product"].split())
        os.rename(file_path+x, f"{new_name}.png")


def main():
    
    train_path = "../model_images/train/"
    test_path = ""
    valid_path = ""

    renaiming_images(train_path, "train")



if __name__ == "__main__":
    main()
