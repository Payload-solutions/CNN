

import os
from PIL import Image
from torch import nn
import torch
import imageio

BASE_DIR = "images/"


def read_image_as_tensor():
    
    dir_images = [x for x in os.listdir(BASE_DIR)]
    
    for x in dir_images:
        image = imageio.imread((os.path.join(BASE_DIR, x)))
        print(torch.from_numpy(image).float(), "\n")
        tensor_image = torch.from_numpy(image).float() / 255

        print(tensor_image)
        break



def main():
    read_image_as_tensor()


if __name__ == "__main__":
    main()
