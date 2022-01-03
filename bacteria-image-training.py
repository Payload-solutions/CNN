

# tensorflow tools
from tensorflow.keras import (
    regularizers,
    models,
    optimizers,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Conv2D,
    MaxPooling2D,
    Activation,
    BatchNormalization,
    Dropout,
    Flatten
)
from tensorflow.keras.callbacks import (ModelCheckpoint)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# data typing handler
from typing import (
    Tuple,
    Any
)


def image_set_generator(train_set: str,
                        test_set: str,
                        validator_set: str) -> Tuple[Any, Any]:
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

    # performing train generator for the neural net
    train_generator = train_gen.flow_from_directory(
        train_set,
        target_size=(64, 64),
        batch_size=16,
        class_mode="categorical"
    )
    valid_gen = valid_gen.flow_from_directory(
        validator_set,
        target_size=(64, 64),
        batch_size=16,
        class_mode="categorical"
    )
    return train_generator, valid_gen


def defining_model() -> Any:
    model = Sequential()

    # first convolution
    model.add(Conv2D(
        filters=32,
        kernel_size=(3, 3),
        padding="same",
        kernel_regularizer=regularizers.l2(1e-4),
        input_shape=(64, 64, 3)
    ))
    model.add(Activation("relu"))
    # model.add(BatchNormalization())

    # second convolution
    model.add(Conv2D(
        filters=32,
        kernel_size=(3, 3),
        padding="same",
        kernel_regularizer=regularizers.l2(1e-4)
    ))
    model.add(Activation("relu"))
    # model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    # third convolution, more  
    model.add(Conv2D(
        filters=64,
        kernel_size=(3, 3),
        padding="same",
        kernel_regularizer=regularizers.l2(1e-4)
    ))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    # fourth convolution
    model.add(Conv2D(
        filters=64,
        kernel_size=(3, 3),
        padding="same",
        kernel_regularizer=regularizers.l2(1e-4)
    ))
    model.add(Activation("relu"))
    # model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    # fifth convolution
    model.add(Conv2D(
        filters=64,
        kernel_size=(3, 3),
        padding="same",
        kernel_regularizer=regularizers.l2(1e-4)
    ))
    model.add(Activation("relu"))
    model.add(BatchNormalization())

    # sixth convolution
    model.add(Conv2D(
        filters=64,
        kernel_size=(3, 3),
        padding="same",
        kernel_regularizer=regularizers.l2(1e-4)
    ))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    # classification flatten
    model.add(Flatten())
    model.add(Dense(2, activation="softmax"))
    model.summary()

    return model


def training():
    train_generator, valid_generator = image_set_generator(train_set="image_set/train",
                        test_set="image_set/test",
                        validator_set="image_set/validator")
    model = defining_model()

    # making checkpoint
    checkpoint = ModelCheckpoint("bacteria_trained.hdf5", 
                             monitor="val_accuracy", 
                             verbose=1, 
                             save_best_only=True)

    model.compile(loss="categorical_crossentropy", 
              optimizer=optimizers.Adam(), 
              metrics=["accuracy"])
    
    hist = model.fit(train_generator, 
                 steps_per_epoch=531//32, 
                 epochs=100, 
                 validation_data=valid_generator,
                 validation_steps=76//32,
                 callbacks=[checkpoint])
    
    return hist


def main():
    hist = training()


if __name__ == "__main__":
    main()
