
import re
import os
from PIL import (
    Image,
    ImageEnhance
)
from torch import nn
import torch
import imageio
import os
from pprint import pprint
from torch.utils.data import (
    Dataset,
    DataLoader
)
from pprint import pprint
import pandas as pd

"""
Convert and resize the pixels of every image.
the new images generated and converted, will be 
stored in the temporal folder
"""


def resize_images():
    base_dir: str = "datasets/test_bacteria/"
    new_dir: str = "temp/"

    files = [x for x in os.listdir(base_dir)]
    for pos, x in enumerate(files):
        image = Image.open(os.path.join(base_dir, x))
        new_image = image.rotate(90)
        new_image.save(new_dir+f"{pos+83}_bacteria.png")
        # print(torch.from_numpy(imageio.imread(os.path.join("datasets/test_bacteria/", x))).float().shape)



def image_handling():

    base_dir = "backup/"
    new_dir = "temp/"
    
    files = [value for value in os.listdir(base_dir)]
    print(len(files))
    for pos, file in enumerate(files):
        image = Image.open(os.path.join(base_dir, file))
        new_image = image.resize((64,64))
        new_image.save(new_dir+f"{pos}_bacteria.png")


def matching_shapes():

    test_files = [file for file in os.listdir("temp/image_train")]

    for x in test_files:
        read_image = imageio.imread(os.path.join("temp/image_train", x))
        # print(torch.from_numpy(read_image).float().shape)
        if torch.from_numpy(read_image).float().shape == (64,64,4):
            print(x)
            # image = Image.open(os.path.join("temporal/", x))
            # image.save(f"backup/{x}")


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
    base_dir = "datasets/image_train"
    files = os.listdir(base_dir)
    files.sort()
    key_values = [re.search(r'\d{1,4}', x).group() for x in files if re.search(r'\d{1,4}',x)]
    dataset = pd.read_csv("datasets/milk-properties.csv")
    print("[*] Dataset finished created successfully...")
    dataset.iloc[key_values[:]].to_csv("datasets/image_train/train-milk-properties.csv", index=False)




def dataset_helper():

    print("\n\t[*] Intializing converter to csv from image properties\n\n")
    dataset = pd.read_csv("datasets/milk-properties.csv")
    files_names = [x for x in os.listdir("temp/image_train/")]
    pprint(files_names)
    targets_num = [int(re.search(r'\d{1,4}', x).group()) for x in files_names]
    # print(dataset)
    # print(sorted(targets_num))

    targets_num.sort()
    dataset.iloc[targets_num[:]].to_csv("train-milk-properties.csv", index=False)
    print("\n\t[*] Finished converter...\n\n")



def main():
    # image_handling()
    # resize_images()
    # matching_shapes()
    dataset_helper()
    # files = [file for file in os.listdir("datasets/test_bacteria/")]
    # for x in files:
    #     print(torch.from_numpy(imageio.imread(os.path.join("datasets/test_bacteria/", x))).float().shape)
    # resize_images()
    



if __name__ == "__main__":
    main()
