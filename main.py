
# import warnings
import os
from PIL import (
    Image,
    ImageEnhance
)
from torch import nn
import torch
import imageio
import os
from pprint import pprint
from torch.utils.data import (
    Dataset,
    DataLoader
)


"""
Convert and resize the pixels of every image.
the new images generated and converted, will be 
stored in the temporal folder
"""


def resize_images():
    base_dir: str = "datasets/test_bacteria/"
    new_dir: str = "temp/"

    files = [x for x in os.listdir(base_dir)]
    for pos, x in enumerate(files):
        image = Image.open(os.path.join(base_dir, x))
        new_image = image.rotate(90)
        new_image.save(new_dir+f"{pos+83}_bacteria.png")
        # print(torch.from_numpy(imageio.imread(os.path.join("datasets/test_bacteria/", x))).float().shape)



def image_handling():

    base_dir = "backup/"
    new_dir = "temp/"
    
    files = [value for value in os.listdir(base_dir)]
    print(len(files))
    for pos, file in enumerate(files):
        image = Image.open(os.path.join(base_dir, file))
        new_image = image.resize((64,64))
        new_image.save(new_dir+f"{pos}_bacteria.png")


def matching_shapes():

    test_files = [file for file in os.listdir("temp/image_train")]

    for x in test_files:
        read_image = imageio.imread(os.path.join("temp/image_train", x))
        # print(torch.from_numpy(read_image).float().shape)
        if torch.from_numpy(read_image).float().shape == (64,64,4):
            print(x)
            # image = Image.open(os.path.join("temporal/", x))
            # image.save(f"backup/{x}")


def main():
    # image_handling()
    # resize_images()
    matching_shapes()

    # files = [file for file in os.listdir("datasets/test_bacteria/")]
    # for x in files:
    #     print(torch.from_numpy(imageio.imread(os.path.join("datasets/test_bacteria/", x))).float().shape)
    # resize_images()
    



if __name__ == "__main__":
    main()
