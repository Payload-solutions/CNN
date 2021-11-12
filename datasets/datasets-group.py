"""To make possible the asignation of every value into the
datasets of train and test, it's neccessary to match single values
with values of the numeric dataset"""


from pprint import pprint
import imageio
from PIL import Image
import os


"""def asign_values_images():
    base_dir = "train_bacteria/"
    locate_dir = "image_train"
    files = os.listdir(base_dir)

    for num, x in enumerate(files):
        image = Image.open(os.path.join(base_dir, x))
        # print(image.format)
        image.save(f'{locate_dir}/{num}_bacteria.png')"""



def load_image_size():
    base_dir = "test_bacteria/"
    locate_dir = "image_test"
    files = os.listdir(base_dir)

    for num, x in enumerate(files):
        image = Image.open(os.path.join(base_dir, x))
        # print(image.size)
        new_image = image.resize((64,64))
        new_image.save(f'{locate_dir}/{num}_bacteria.png')



def match_values_images():
    pass


if __name__ == "__main__":
    # asign_values_images()
    load_image_size()
