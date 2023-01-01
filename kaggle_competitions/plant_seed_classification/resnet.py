# Import the necessary packages
import os
import numpy as np

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Conv2D, Activation, Flatten, Dense, Dropout, BatchNormalization, MaxPooling2D
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD, Adam
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from keras.callbacks import LearningRateScheduler, EarlyStopping
from keras.callbacks import ModelCheckpoint
from math import exp
from keras.preprocessing.image import ImageDataGenerator
# from keras.applications.resnet_v2 import preprocess_input
from keras.applications.convnext import preprocess_input


def main():
    data_root = r'C:\kaggle\plant_seedling_classification\plant-seedlings-classification\train'
    batch_size = 4
    seed = 42
    val_split = 0.2
    image_size = (320, 320)

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,  # Standandardize for Resnet
        rotation_range=30,  # Int. Degree range for random rotations.
        zoom_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        brightness_range=[0.7, 1.3],
        rescale=0.9,
        vertical_flip=True,
        horizontal_flip=True,
        validation_split=val_split)

    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=val_split)

    train_generator = train_datagen.flow_from_directory(
        data_root,  # It should contain one subdirectory per class.
        target_size=image_size,  # Dims to which all images found will be resized.
        color_mode='rgb',
        batch_size=batch_size,
        class_mode="categorical",  # Type of label arrays that are returned
        subset='training',
        seed=seed,
        shuffle=True
    )

    val_generator = val_datagen.flow_from_directory(
        data_root,
        target_size=image_size,
        color_mode='rgb',
        batch_size=batch_size,
        class_mode="categorical",
        subset='validation',
        seed=seed,
        shuffle=True
    )

    input_shape_c = tuple((image_size[0], image_size[1], 3))
    print(input_shape_c)

    base_model = ResNet50V2(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape_c)  # It should have exactly 3 inputs channels, and width and height should be no smaller than 32.

    base_model = tf.keras.applications.convnext.ConvNeXtBase(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape_c,
    )
    # base_model.summary()
    # We freeze layers in first 4 convolutional blocks. Fifth will be re trained
    for layer in base_model.layers:
        if layer.name == 'conv5_block1_1_conv':
            break
        layer.trainable = False

        # We add more layers to complete the classification part
    pre_trained_model = Sequential()
    pre_trained_model.add(base_model)
    pre_trained_model.add(layers.Flatten())

    pre_trained_model.add(layers.Dense(512, activation='relu'))

    pre_trained_model.add(Dropout(0.5))
    pre_trained_model.add(BatchNormalization())

    pre_trained_model.add(layers.Dense(12, activation='softmax'))
    pre_trained_model.summary()

    # We compile the model
    epochs = 50

    print("[INFO]: Compiling the model...")
    pre_trained_model.compile(loss="categorical_crossentropy",
                              optimizer=Adam(learning_rate=1e-3),
                              metrics=["accuracy"])

    def scheduler(epoch, lr):
        if epoch < 10:
            return lr
        else:
            return lr * exp(-0.1)

    annealer = LearningRateScheduler(scheduler)

    earlystop = EarlyStopping(
        patience=5,
        monitor="val_loss",
    )

    model_save = ModelCheckpoint(
        filepath=os.path.join(r'C:\kaggle\plant_seedling_classification\models', 'convnext_base.h5'),
        save_best_only=True,
        monitor='val_accuracy',
        verbose=1)

    # training the data on the model
    print("[INFO]: Beginning the training ...")
    history_pretrained = pre_trained_model.fit(
        train_generator,
        validation_data=val_generator,
        # steps_per_epoch=train_generator.n // train_generator.batch_size,
        # validation_steps=val_generator.n // val_generator.batch_size,
        epochs=epochs,
        callbacks=[model_save, annealer],
    )


if __name__ == '__main__':
    main()
