
import matplotlib.pyplot as plt
import numpy as np
from skimage import (
    io,
    color
)
import os
import scipy.ndimage as nd


if __name__ == "__main__":
    
    list_images = [x for x in os.listdir("../datasets/temp/image_train/")]

    for x in list_images:

        image = io.imread(os.path.join("../datasets/temp/image_train/", x))

        print(image.shape)

        # transforming to gray scale

        image_gray = color.rgb2gray(image)
        print(image_gray)
        print(image_gray.shape)

        kernel = np.array([
            [-1, 0, 1],
            [-1, 0, 1],
            [-1, 0, 1]])

        new_image = nd.convolve(image_gray, kernel)

        # print(new_image)
        # print(new_image.shape)

        fix, axes = plt.subplots(1,2, figsize=(15,10))
        axes[0].imshow(image_gray, cmap=plt.cm.gray)
        axes[0].set_title('Original')
        axes[0].axis('off')

        axes[1].imshow(new_image, cmap=plt.cm.gray)
        axes[1].set_title('Convolution')
        axes[1].axis('off')

        plt.show()


        break