import os
# os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2

# Import the necessary packages
import numpy as np
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
from keras.applications.resnet_v2 import preprocess_input

from modeling import ModelCreator


def main():
    print('train plant seed classifier')
    data_root = r'C:\kaggle\plant_seedling_classification\plant-seedlings-classification\train'
    seed_types = os.listdir(data_root)
    print(seed_types)
    seed = 42

    image_size = 224
    batch_size = 4
    val_split = 0.2

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
        target_size=(image_size, image_size),  # Dims to which all images found will be resized.
        color_mode='rgb',
        batch_size=batch_size,
        class_mode="categorical",  # Type of label arrays that are returned
        subset='training',
        seed=seed,
        shuffle=True
    )

    val_generator = val_datagen.flow_from_directory(
        data_root,
        target_size=(image_size, image_size),
        color_mode='rgb',
        batch_size=batch_size,
        class_mode="categorical",
        subset='validation',
        seed=seed,
        shuffle=True
    )

    input_shape_c = tuple((image_size, image_size, 3))
    print(input_shape_c)

    base_model = ResNet50V2(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape_c)

    # We add more layers to complete the classification part
    pre_trained_model = Sequential()
    pre_trained_model.add(base_model)
    pre_trained_model.add(layers.Flatten())

    pre_trained_model.add(layers.Dense(512, activation='relu'))

    pre_trained_model.add(Dropout(0.5))
    pre_trained_model.add(BatchNormalization())

    pre_trained_model.add(layers.Dense(12, activation='softmax'))
    pre_trained_model.summary()

    pre_trained_model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=1e-3),
        metrics=["accuracy"]
    )

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    dst_dir = r'C:\kaggle\plant_seedling_classification\plant-seedlings-classification\model'
    train_dir = os.path.join(dst_dir, 'convnext_tiny')
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(train_dir, 'model.h5'),
        save_best_only=True,
        monitor='val_accuracy',
        verbose=1
    )
    board_callback = tf.keras.callbacks.TensorBoard(
        os.path.join(train_dir, 'logs'),
    )
    print(train_generator.class_indices)

    result = pre_trained_model.fit_generator(
        train_generator,
        epochs=50,
        validation_data=val_generator,
        callbacks=[early_stop, cp_callback, board_callback],
        verbose=1
    )
    print('finished plant seed classifier')


if __name__ == '__main__':
    main()
