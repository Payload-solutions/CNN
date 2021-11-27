

import torch
import imageio
from PIL import Image
import os


def handler_image():

    base_dir = "datasets/image_train/"
    # base_dir = "backup/"
    new_dir = ""
    files = [file for file in os.listdir(base_dir)]
    
    for x in files:
        if torch.from_numpy(imageio.imread(os.path.join(base_dir, x))).float().shape == (64,64,3):
            image_read = Image.open(os.path.join(base_dir, x))
            image_read.save("image/"+x)



def main():

    base_dir = "image/models"
    new_dir = "image/image_train/"

    files = [x for x in os.listdir(base_dir)]

    for pos, x in enumerate(files):
        image = Image.open(os.path.join(base_dir, x))
        image.save(new_dir+f"{pos+70}_bacteria.png")




if __name__ == "__main__":
    main()
