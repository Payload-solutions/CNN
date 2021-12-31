# First test case for the convolutional neural networks
# using tensorflow
# @author: Arturo Negreiros Pxyl0xd


import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import (
    Conv2D,
    Dropout,
    MaxPooling2D,
    Flatten,
    Dense
)
import numpy as np
import matplotlib.pyplot as plt



def main():
    (train_image, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    
    # normalizing images
    train_image = train_image.astype('float32')/255
    test_images = test_images.astype('float32')/255

    # reshaping the images
    train_image = train_image.reshape(train_image.shape[0], 28,28,1)
    test_images = test_images.reshape(test_images.shape[0], 28,28,1)

    print(train_labels[9])

    train_labels = tf.keras.utils.to_categorical(train_labels, 10)
    test_labels = tf.keras.utils.to_categorical(test_labels, 10)

    # crating model convolutional
    model = tf.keras.Sequential()

    model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.3))

    model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    model.summary()


    # compile and training

    model.compile(loss='categorical_crossentropy',
        optimizer='rmsprop',
        metrics=['accuracy']
    )

    # model.fit(train_image, train_labels, batch_size=64, epochs=10)

    print(model.evaluate(test_images, test_labels, verbose=0))


    # if this metric not start to growth, stop training
    # this si influenced by the training
    # early = tf.keras.callbacks.EarlyStopping(monitor="accuracy", patience=1)

    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath="convolutional_weights.hdf5", 
        verbose=1, 
        monitor="accuracy", 
        save_best_only=True)

    # model.fit(train_image, train_labels, batch_size=64,callbacks=[checkpoint],epochs=10)

    # After training the model
    # we gonna load the weights using the load_weights function
    # without need to make another training again

    model2 = model
    model2.load_weights("./convolutional_weights.hdf5")

    history_data = model2.evaluate(test_images, test_labels)


    print(type(history_data))

    print(dir(history_data))

    
    
    print(f"In the previous training, we can be sure that the accuracy is the {float(history_data[1])*100}")

if __name__ == "__main__":
    main()
