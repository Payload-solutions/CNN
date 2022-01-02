
import os
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
import re
import shutil


def main():
    data_train = "image_set/train/"
    new_dir = "image_set/"
    files = os.listdir(data_train)
    for x in files:
        src = re.findall(r'(.*)\.\d{1,5}.png', x)[0]
        if src == "":
            pass
        elif src == "":
            pass
        else:
            pass
        print(src)
    # shutil.move("verga.txtr", new_dir+"Non_fat_yogurt/")
    # FILE_PATH = "/train"
    # train_path = FILE_PATH + "train"
    # test_path = FILE_PATH+"test"
    # valid_path = FILE_PATH+"validator"
    # filter_base = 32
    # w_regualizers = 1e-4
    # train_gen = ImageDataGenerator(
    #     rescale=1./255,
    #     rotation_range=40,
    #     width_shift_range=0.2,
    #     height_shift_range=0.2,
    #     shear_range=0.2,
    #     horizontal_flip=True
    # )

    # test_gen = ImageDataGenerator(
    #     rescale=1./255
    # )

    # valid_gen = ImageDataGenerator(
    #     rescale=1./255
    # )
    # train_generator = train_gen.flow_from_directory(train_path, target_size=(64,64), batch_size=32, class_mode="categorical")
    # test_generator = test_gen.flow_from_directory(valid_path, target_size=(150,150), batch_size=32, class_mode="categorical")


if __name__ == "__main__":
    main()
