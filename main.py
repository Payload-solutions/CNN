
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import re
import shutil

def permuting_directories():
    # # data_train = "image_set/train/"
    # # data_test = "image_set/test/"
    # data_valid = "image_set/validator/"
    # new_dir = "image_set/"
    # files = os.listdir(data_valid)
    # for x in files:
    #     src = re.findall(r'(.*)\.\d{1,5}.png', x)[0]
    #     if src == "Regular_yogurt":
    #         shutil.move(data_valid+x, new_dir+f"{src}/{x}")
    #     elif src == "Low_fat_yogurt":
    #         shutil.move(data_valid+x, new_dir+f"{src}/{x}")
    #     else:
    #         shutil.move(data_valid+x, new_dir+f"{src}/{x}")
    # print("Done!!!")
    # shutil.move("verga.txtr", new_dir+"Non_fat_yogurt/")
    pass

def main():
    
    FILE_PATH = "image_set/"
    train_path = FILE_PATH + "train"
    test_path = FILE_PATH+"test"
    valid_path = FILE_PATH+"validator"
    filter_base = 32
    w_regualizers = 1e-4
    train_gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        horizontal_flip=True
    )

    test_gen = ImageDataGenerator(
        rescale=1./255
    )

    valid_gen = ImageDataGenerator(
        rescale=1./255
    )
    train_generator = train_gen.flow_from_directory(train_path, target_size=(64,64), batch_size=32, class_mode="categorical")
    test_generator = test_gen.flow_from_directory(valid_path, target_size=(150,150), batch_size=32, class_mode="categorical")


if __name__ == "__main__":
    main()
