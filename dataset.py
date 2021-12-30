

from typing import Tuple
import pandas as pd
import numpy


class Dataset:

    def __init__(self, csv_dir: str, train_range=528, test_range=236) -> None:
    
        self.train_range = train_range
        self.test_range = test_range
        self.dataset = pd.read_csv(csv_dir)

    def spliting_dataset(self):

        target_train = None
        target_test = None



