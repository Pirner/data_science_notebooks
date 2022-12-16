import os
# os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
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

    image_size = 224
    batch_size = 2

    image_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        # rescale=1. / 255,
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
        batch_size=batch_size,
    )
    validation_generator = image_gen.flow_from_directory(
        directory=data_root,
        target_size=(image_size, image_size),
        color_mode='rgb',
        class_mode="categorical",
        subset='validation',
        batch_size=batch_size,
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

    # model = ModelCreator.create_base_model(image_size=image_size)
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    preprocess_input = tf.keras.applications.convnext.preprocess_input
    rescale = tf.keras.layers.Rescaling(1. / 127.5, offset=-1)

    base_model = tf.keras.applications.convnext.ConvNeXtSmall(
        input_shape=(image_size, image_size, 3),
        include_top=False,
        weights='imagenet'
    )

    image_batch, label_batch = next(iter(train_generator))
    feature_batch = base_model(image_batch)
    print(feature_batch.shape)
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    feature_batch_average = global_average_layer(feature_batch)
    print(feature_batch_average.shape)

    prediction_layer = tf.keras.layers.Dense(12)
    prediction_batch = prediction_layer(feature_batch_average)
    print(prediction_batch.shape)

    inputs = tf.keras.Input(shape=(image_size, image_size, 3))
    # x = data_augmentation(inputs)
    x = preprocess_input(inputs)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)
    model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # model = ModelCreator.create_resnet50_model(image_size=image_size)
    # exit(0)
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

    result = model.fit_generator(
        train_generator,
        epochs=50,
        validation_data=validation_generator,
        callbacks=[early_stop, cp_callback, board_callback],
        verbose=1
    )
    print('finished plant seed classifier')


if __name__ == '__main__':
    main()
