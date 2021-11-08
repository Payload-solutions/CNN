"""To make possible the asignation of every value into the
datasets of train and test, it's neccessary to match single values
with values of the numeric dataset"""


import pandas as pd
import numpy as np
import imageio
from PIL import Image

BASE = "milk-properties.csv"

def asign_values_images():
    data = pd.read_csv(BASE)
    print(data)


if __name__ == "__main__":
    asign_values_images()
