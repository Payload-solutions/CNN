
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import os


# image dir
BASE_DIR = "../datasets/temp/image_train/"


def handler():
    
    list_images = [x for x in os.listdir(BASE_DIR)]
    # print(list_images)

    for x in list_images:
        image = io.imread(os.path.join(BASE_DIR, x))
        
        r = image[:, :, 0]
        g = image[:, :, 1]
        b = image[:, :, 2]

        print(r.shape)
        print(g.shape)
        print(b.shape)

        aux_dim = np.zeros([64,64])

        new_r = np.dstack((r, aux_dim, aux_dim)).astype(np.uint8)
        new_g = np.dstack((aux_dim, g, aux_dim)).astype(np.uint8)
        new_b = np.dstack((aux_dim, aux_dim, b)).astype(np.uint8)
        print(new_r.shape)        
        plt.imshow(new_r)
        plt.show()

        """
        to make negative image:

            negativ = 255 - image

        """
        break

def main():
    handler()



if __name__ == "__main__":
    main()