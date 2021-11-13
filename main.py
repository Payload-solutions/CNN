
import warnings
import os
from PIL import Image
from torch import nn
import torch
import imageio
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# In detail:-
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed

"""BASE_DIR = "images/"


def read_image_as_tensor():
    
    dir_images = [x for x in os.listdir(BASE_DIR)]
    
    for x in dir_images:
        image = imageio.imread((os.path.join(BASE_DIR, x)))
        print(torch.from_numpy(image).float(), "\n")
        tensor_image = torch.from_numpy(image).float() / 255

        print(tensor_image)
        break


"""
def main():
    # read_image_as_tensor()
    print(tf.__version__)


if __name__ == "__main__":
    main()
