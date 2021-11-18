
# import warnings
import os
from PIL import Image
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
def image_handling():
    pass







class ImageDataset(nn.Module):
    def __init__(self):
        super(ImageDataset, self).__init__()


def main():
    # read_image_as_tensor()

    # image_dataset = ImageDataset()
    
    image_files = [file for file in os.listdir("64x64_SIGNS/train_signs/")]
    another_files = [file for file in os.listdir("datasets/image_train/")]

    test_files = [file for file in os.listdir("datasets/test_bacteria/")]

    for x in test_files:
        read_image = imageio.imread(os.path.join("datasets/test_bacteria", x))
        if torch.from_numpy(read_image).float().shape == (64,64,3):
            print(x)

    for x in another_files:
        read_image = imageio.imread(os.path.join("datasets/image_train/", x))
        if torch.from_numpy(read_image).float().shape == (64,64,3):
            print(x)

    print("\n\n\n")
    for pos, x in enumerate(image_files):
        read_image = imageio.imread(os.path.join("64x64_SIGNS/train_signs/", x))
        print(torch.from_numpy(read_image).float().shape)
        
        if pos == 100:
            break

if __name__ == "__main__":
    main()
