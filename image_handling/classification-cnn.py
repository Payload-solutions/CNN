"""
Simply classification problem solving
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
    Activation
)
from tensorflow.keras.datasets import cifar10

# for data-cleaning
import numpy as np
import matplotlib.pyplot as plt


def defining_parameters():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # datacleaning
    ## first of all, reduce the dimensionship
    ## of our matrix, come from 0 to 255

    x_train = x_train.astype('float32')/255
    x_test = x_test.astype('float32')/255

    num_classes = len(np.unique(y_train))

    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    # another good practice it's split the set of dataset
    ## trainin, test and validation

    (x_train, x_valid) = x_train[5000:], x_train[:5000]
    (y_train, y_valid) = y_train[5000:], y_train[:5000]


    # defining model
    filter_base = 32
    w_regularizers = 1e-4

    ## building model
    model = Sequential()
    
    # 1 convolution
    model.add(Conv2D(filter_base, kernel_size=(3,3), padding='same', kernel_regularizer=regularizers.l2(w_regularizers), input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    
    # 2 convolution
    model.add(Conv2D(filter_base, kernel_size=(3,3), padding='same', kernel_regularizer=regularizers.l2(w_regularizers)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))


    # 3 convolution
    # this convolution, the layer gonna be more deep
    model.add(Conv2D(2*filter_base, kernel_size=(3,3), padding='same', kernel_regularizer=regularizers.l2(w_regularizers)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    # 4 convolution
    model.add(Conv2D(2*filter_base, kernel_size=(3,3), padding='same', kernel_regularizer=regularizers.l2(w_regularizers)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.3))

    # 5 convolution
    model.add(Conv2D(2*filter_base, kernel_size=(3,3), padding='same', kernel_regularizer=regularizers.l2(w_regularizers)))
    model.add(Activation('relu'))


    # 6 convolution
    model.add(Conv2D(2*filter_base, kernel_size=(3,3), padding='same', kernel_regularizer=regularizers.l2(w_regularizers)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))


    ## Classification - flatten
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))

    
    model.summary()


    # compiling the model
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', 
            metrics=['accuracy'])

    hist = model.fit(x_train, y_train, batch_size=32, epochs=100, validation_data=(x_valid, y_valid), verbose=2, shuffle=True)

    print(hist)



def main():
    defining_parameters()



if __name__ == "__main__":
    main()
