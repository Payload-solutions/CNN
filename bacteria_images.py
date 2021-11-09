"""Class for testing the architecture"""

import re
from pprint import pprint
import os
from os.path import join
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import (
    Dataset,
    DataLoader
)
from PIL import Image
import pandas as pd



class Net(nn.Module):
    def __init__(self, num_channels: int):
        super(Net, self).__init__()

        self.num_channels = num_channels

        # this network, gotta have 3 layers

        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=self.num_channels,
                               kernel_size=3, stride=1,
                               padding=1)

        self.conv2 = nn.Conv2d(in_channels=self.num_channels,
                               out_channels=self.num_channels * 2,
                               kernel_size=3,
                               stride=1, padding=1)

        self.conv3 = nn.Conv2d(in_channels=self.num_channels * 2,
                               out_channels=self.num_channels * 4,
                               kernel_size=3,
                               stride=1, padding=1)

        # full connected
        # max pool type: reduce by 2 the size of the layer
        self.fc1 = nn.Linear(in_features=self.num_channels * 4 * 8 * 8,
                             out_features=self.num_channels * 4)

        self.fc2 = nn.Linear(in_features=self.num_channels * 4,
                             out_features=1)

    def forward(self, x):
        """we start with a image of 3 channels 64x64 pixels"""

        x = self.conv1(x)  # num_channels 64x64
        x = F.relu(F.max_pool2d(x, 2))  # num_channels with 32x32

        x = self.conv2(x)  # num_channels*2 32x32
        x = F.relu(F.max_pool2d(x, 2))  # num_channels*2 16x16

        x = self.conv3(x)  # num_channels*4 16x16
        x = F.relu(F.max_pool2d(x, 2))  # num_channels*4 8x8

        # flatten
        x = x.view(-1, self.num_channels * 4 * 8 * 8)

        # fully connected
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.softmax(x, dim=1)

        return x


class BacteriaDataset(Dataset):
    def __init__(self, base_dir: str, split="train", transform=None) -> None:
        super(BacteriaDataset, self).__init__()
        path = join(base_dir, f"{split}_bacteria")
        files = os.listdir(path)

        self.filenames = [join(path, file) for file in files if file.endswith(".png")]
        # self.targets = [int(f[0]) for f in files]
        print(files)
        # self.transform = transform
        pprint(self.filenames)
    """def __len__(self):
        return len(self.filenames)

    def __getitem__(self, item):
        image = Image.open(self.filenames[item])

        if self.transform:
            image = self.transform(image)
        return image, self.targets[item]
    """



def load_numeric_dataset():
    r"""We gonna make possible the permutation of the image name
    to can match with the numeric dataset
    getting the first match using iloc, we can get the first row

    [
    'streptococcus_initial_strain_cfu_ml'
    'lactobacillus_initial_strain_cfu_ml' 
    'ideal_temperature_c'
    'minimum_milk_proteins'
    'titratable_acidity' 
    'pH_milk_sour_'
    'fat_milk_over_100mg_' 
    'quality_product'
    ]
    
    >>> dataset.iloc[0].to_numpy() => :
    [4.693 5.376 40.593 2.622 1.171 4.539 1.8156 'Low fat yogurt']

    """
    base_dir = "datasets/train_bacteria"
    files = os.listdir(base_dir)
    files.sort()
    key_values = [re.search(r'\d{1,4}', x).group() for x in files if re.search(r'\d{1,4}',x)]
    dataset = pd.read_csv("datasets/milk-properties.csv")
    print("[*] Dataset finished created successfully...")
    dataset.iloc[key_values[:]].to_csv("datasets/train_bacteria/train-milk-properties.csv", index=False)


def main():
    # bacteria = BacteriaDataset(base_dir="images")
    load_numeric_dataset()


if __name__ == "__main__":
    main()
