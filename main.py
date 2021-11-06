

import os
from PIL import Image
from torch import nn
from pprint import pprint
class Dataset(nn.Module):
    pass


BASE_DIR = "images/"


def load_images():
    
    dir_images = [x for x in os.listdir(BASE_DIR)]
    pprint(len(dir_images))



def main():
    load_images()


if __name__ == "__main__":
    main()
