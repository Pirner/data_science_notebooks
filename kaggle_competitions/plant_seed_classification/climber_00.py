import os

import tensorflow as tf
import pandas as pd


data_root = r'C:\kaggle\plant_seedling_classification\plant-seedlings-classification'
train_val_dir = os.path.join(data_root, 'train')
test_dir = os.path.join(data_root, 'test')


def define_callbacks():
    save_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='model.h5',
        monitor='val_acc',
        save_best_only=True,
        verbose=1
    )

    return save_callback


def define_model(width, height):
    model_input = tf.keras.layers.Input(shape=(width, height, 3), name='image_input')
    model_main = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=False, weights='imagenet')(model_input)
    model_dense1 = tf.keras.layers.Flatten()(model_main)
    model_dense2 = tf.keras.layers.Dense(128, activation='relu')(model_dense1)
    model_out = tf.keras.layers.Dense(12, activation="softmax")(model_dense2)

    model = tf.keras.models.Model(model_input,  model_out)
    optimizer = tf.keras.optimizers.Adam(lr=0.00004, beta_1=0.9, beta_2=0.999)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model


def define_generators(width, height, batch_size):
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=360,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.5,
        vertical_flip=True,
        horizontal_flip=True,
        validation_split=0.2,  # change to use validation instead of training on entire training set
    )

    train_generator = train_datagen.flow_from_directory(
        directory=train_val_dir,
        target_size=(width, height),
        batch_size=batch_size,
        color_mode='rgb',
        class_mode="categorical",
        subset='training',
    )

    validation_generator = train_datagen.flow_from_directory(
        directory=train_val_dir,
        target_size=(width, height),
        batch_size=batch_size,
        color_mode='rgb',
        class_mode="categorical",
        subset='validation',
    )

    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator()

    test_generator = test_datagen.flow_from_directory(
        directory=data_root,
        classes=['test'],
        target_size=(width, height),
        batch_size=1,
        color_mode='rgb',
        shuffle=False,
        class_mode='categorical')

    return train_generator, validation_generator, test_generator


def main():
    batch_size = 4
    seed = 42
    val_split = 0.2
    image_size = (256, 256)

    nb_epoch = 40
    batch_size = 4
    width = 288
    height = 288
    species_list = ["Black-grass", "Charlock", "Cleavers", "Common Chickweed", "Common wheat", "Fat Hen",
                    "Loose Silky-bent", "Maize", "Scentless Mayweed", "Shepherds Purse", "Small-flowered Cranesbill",
                    "Sugar beet"]

    model = define_model(width, height)
    model.summary()
    train_generator, validation_generator, test_generator = define_generators(
        width=width,
        height=height,
        batch_size=batch_size
    )
    save_callback = define_callbacks()

    model.fit(
        train_generator,
        epochs=nb_epoch,
        steps_per_epoch=train_generator.samples // batch_size,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size,
        callbacks=[save_callback]  # UNCOMMENT THIS LINE TO SAVE BEST VAL_ACC MODEL
    )


if __name__ == '__main__':
    main()
