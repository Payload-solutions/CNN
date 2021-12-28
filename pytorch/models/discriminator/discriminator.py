"""Discriminator module
The goal for discriminator module:
    Take an image and ouput wether or not it is
    a real training image or fake image from the 
    generator

    Discriminative network gonna send numbers int 
    the output by the flatten
"""

from torch import nn
from torch.nn import functional as F


class Discriminator(nn.Module):
    
    def __init__(self, input_nc: int):
        super(Discriminator, self).__init__()

