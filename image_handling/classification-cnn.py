"""
Simply classification problem solving
"""


"""
This is the kind of typically architecture for neural networks
"""



# utilities for deep model
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, 
    MaxPooling2D, 
    Flatten, 
    Dense, 
    Dropout,
    Activation,
    BatchNormalization
)
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    ModelCheckpoint
)


# for data-cleaning
import numpy as np
import matplotlib.pyplot as plt


def defining_parameters():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    print(len(x_train))
    print(y_train)
    print(len(y_train))
    """
        In this case, the number of the x_train and y_train
        is the same. The y_train contain values significatives 
        of each image, in our project, the targets it's the value
        of the final sustrate lactic ["Regular", "Medium", "Low"]
    """
    
    for x in y_train:
        print(x)
        break
    # datacleaning
    ## first of all, reduce the dimensionship
    ## of our matrix, come from 0 to 255

    x_train = x_train.astype('float32')/255
    x_test = x_test.astype('float32')/255

    num_classes = len(np.unique(y_train))

    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    # Implementing of Normalization

    mean = np.mean(x_train)
    std = np.std(x_train)

    x_train = (x_train - mean) / (std + 1e-7)
    x_test = (x_test - mean) / (std + 1e-7)

    # another good practice it's split the set of dataset
    # trainin, test and validation

    (x_train, x_valid) = x_train[5000:], x_train[:5000]
    (y_train, y_valid) = y_train[5000:], y_train[:5000]


    # defining model
    filter_base = 32
    w_regularizers = 1e-4

    ## building model
    model = Sequential()

    # Architectura
    # each architecture layer, we gonna add 
    # a normalization layer.
    
    # 1 convolution
    model.add(Conv2D(filter_base, kernel_size=(3,3), padding='same', kernel_regularizer=regularizers.l2(w_regularizers), input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    
    # 2 convolution
    model.add(Conv2D(filter_base, kernel_size=(3,3), padding='same', kernel_regularizer=regularizers.l2(w_regularizers)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))


    # 3 convolution
    # this convolution, the layer gonna be more deep
    model.add(Conv2D(2*filter_base, kernel_size=(3,3), padding='same', kernel_regularizer=regularizers.l2(w_regularizers)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    # 4 convolution
    model.add(Conv2D(2*filter_base, kernel_size=(3,3), padding='same', kernel_regularizer=regularizers.l2(w_regularizers)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.3))

    # 5 convolution
    model.add(Conv2D(2*filter_base, kernel_size=(3,3), padding='same', kernel_regularizer=regularizers.l2(w_regularizers)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    # 6 convolution
    model.add(Conv2D(2*filter_base, kernel_size=(3,3), padding='same', kernel_regularizer=regularizers.l2(w_regularizers)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))


    ## Classification - flatten
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))

    
    model.summary()

    datagen = ImageDataGenerator(rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True)

    
    # compiling the model
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(), 
            metrics=['accuracy'])

    # hist = model.fit(x_train, y_train, batch_size=32, epochs=100, validation_data=(x_valid, y_valid), verbose=2, shuffle=True)
    checkpoint = ModelCheckpoint("my_best.hdf5", verbose=1, save_best_only=True)
    hist = model.fit(datagen.flow(x_train, y_train), 
        batch_size=128,
        callbacks=[checkpoint],
        steps_per_epoch=x_train.shape[0]//128,
        epochs=120, 
        validation_data=(x_valid, y_valid), 
        verbose=2,
        shuffle=True)
    # print(hist)


    plt.plot(hist.history["accuracy"], label="train")
    plt.plot(hsit.history["val_accuracy"], label="validation")

    plt.legend()
    plt.show()


def main():
    defining_parameters()



if __name__ == "__main__":
    main()
