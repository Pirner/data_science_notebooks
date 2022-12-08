import os
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from modeling import ModelCreator


def main():
    print('train plant seed classifier')
    data_root = r'C:\kaggle\plant_seedling_classification\plant-seedlings-classification\train'
    seed_types = os.listdir(data_root)
    print(seed_types)

    image_size = 244
    batch_size = 2

    image_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.3,
        zoom_range=0.5,
        vertical_flip=True,
        horizontal_flip=True,
        validation_split=0.2,
    )
    train_generator = image_gen.flow_from_directory(
        directory=data_root,
        target_size=(image_size, image_size),
        color_mode='rgb',
        class_mode="categorical",
        subset='training',
    )
    validation_generator = image_gen.flow_from_directory(
        directory=data_root,
        target_size=(image_size, image_size),
        color_mode='rgb',
        class_mode="categorical",
        subset='validation',
    )

    # test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
    # test_generator = test_datagen.flow_from_directory(
    #     directory='/kaggle/input/plant-seedlings-classification/',
    #     classes=['test'],
    #     target_size=(230, 230),
    #     batch_size=1,
    #     color_mode='rgb',
    #     shuffle=False,
    #     class_mode='categorical'
    # )

    model = ModelCreator.create_base_model(image_size=image_size)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    print(train_generator.class_indices)

    result = model.fit_generator(
        train_generator,
        epochs=30,
        steps_per_epoch=3803 // batch_size,
        validation_steps=947 // batch_size,
        validation_data=validation_generator,
        callbacks=[early_stop],
        verbose=1
    )
    print('finished plant seed classifier')


if __name__ == '__main__':
    main()
