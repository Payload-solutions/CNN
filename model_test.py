"""Class for testing the architecture"""

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

def main():
    bacteria = BacteriaDataset(base_dir="images")
    


if __name__ == "__main__":
    main()
