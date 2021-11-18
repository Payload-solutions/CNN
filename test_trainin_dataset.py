import os
from typing import (
    Any,
    Iterable
)
from PIL import Image
import pandas as pd
from torch import nn
from torch.nn import functional as F
import torch
BASE_DIR: str = "datasets/image_train"
BASE_DATASET: str = "datasets/train-milk-properties.csv"
from torch.utils.data import (
    DataLoader,
    Dataset
)
import torchvision.transforms as transforms
from torch import optim


class ImageNet(nn.Module):

    def __init__(self, num_channels: int):
        super(ImageNet, self).__init__()
        self.num_channels = num_channels

        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=self.num_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1)

        self.conv2 = nn.Conv2d(in_channels=self.num_channels,
                               out_channels=self.num_channels * 2,
                               kernel_size=3,
                               stride=1,
                               padding=1)

        self.conv3 = nn.Conv2d(in_channels=self.num_channels * 2,
                               out_channels=self.num_channels * 4,
                               kernel_size=3,
                               stride=1,
                               padding=1)

        self.fc1 = nn.Linear(in_features=self.num_channels * 4 * 8 * 8, out_features=self.num_channels * 4)

        """
        There are only three types of final products to make the measure
        """
        self.fc2 = nn.Linear(in_features=self.num_channels * 4, out_features=3)

    def forward(self, x):
        x = self.conv1(x)  # num_channels x 64 x 64
        x = F.relu(F.max_pool2d(x, 2))  # num_channels x 32 x 32

        x = self.conv2(x)  # num_channels*2 x 32 x 32
        x = F.relu(F.max_pool2d(x, 2))  # num_channels*2 x 16 x 16

        x = self.conv3(x)  # num_channels*4 x 32 x 32
        x = F.relu(F.max_pool2d(x, 2))  # num_channels*4 x 8 x 8

        # flatten
        x = x.view(-1, self.num_channels*4*8*8)

        # fully connected
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.softmax(x, dim=1)

        return x


class ImageDataset(Dataset):
    def __init__(self, base_dir: str, split: str = "train",
                 dataset_base:str = "datasets/train-milk-properties.csv",
                 transform=None):
        path = os.path.join(base_dir, f"image_{split}")
        files = os.listdir(path)

        self.files_names = [os.path.join(path, f) for f in files]
        self.transform = transform
        self.dataset_base = dataset_base
        self.targets = self._load_dataset()

    def _load_dataset(self) -> list:
        return pd.read_csv(self.dataset_base)["quality_product"].to_numpy()

    def __len__(self):
        return len(self.files_names)

    def __getitem__(self, item):
        image = Image.open(self.files_names[item])
        if self.transform:
            image = self.transform(image)
        return image, self.targets[item]


class RunningMetrics:
    def __init__(self):
        self.S = 0
        self.N = 0

    def update(self, val, size):
        self.S += val
        self.N += size
    
    def __call__(self):
        return self.S/float(self.N)


def main():
    image_dataset = ImageDataset(base_dir="datasets", split="train" ,dataset_base=BASE_DATASET, transform=transforms.ToTensor())

    dataloader = DataLoader(image_dataset, batch_size=8)
    # print(len(image_dataset))

    net = ImageNet(32)
    loss_fn = nn.NLLLoss()
    optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)

    num_epochs = 100

    for epoch in range(num_epochs):
        print(f"{epoch}/{num_epochs}")
        print("-"*15)

        running_loss = RunningMetrics()
        running_acc = RunningMetrics()

        for inputs, targets  in dataloader:

            optimizer.zero_grad()

            outputs = net(inputs)

            _, pred = torch.max(outputs, 1)

            loss = loss_fn(outputs, targets)

            loss.backward()
            optimizer.step()

            batch_size = inputs.size()[0]
            running_loss.update(loss.item()*batch_size, batch_size)

            running_acc.update(torch.sum(pred == targets).float(), batch_size)
        
        print(f"Loss {running_loss()}   Acc{running_acc()}" )

if __name__ == "__main__":
    main()
