import tensorflow as tf
import pandas as pd


def main():
    test_data_path = r'C:\kaggle\plant_seedling_classification\plant-seedlings-classification'
    model_path = r'C:\kaggle\plant_seedling_classification\plant-seedlings-classification\model\model.h5'
    data_root = r'C:\kaggle\plant_seedling_classification\plant-seedlings-classification\train'
    image_size = 244

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
        batch_size=2,
    )

    class_labels = list(train_generator.class_indices)

    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
    test_generator = test_datagen.flow_from_directory(
        directory=test_data_path,
        classes=['test'],
        target_size=(image_size, image_size),
        batch_size=1,
        color_mode='rgb',
        shuffle=False,
        class_mode='categorical')

    model = tf.keras.models.load_model(model_path)

    predictions = model.predict(test_generator, steps=test_generator.samples)

    class_list = []

    for i in range(0, predictions.shape[0]):
        y_class = predictions[i, :].argmax(axis=-1)
        class_list += [class_labels[y_class]]

    submission = pd.DataFrame()
    submission['file'] = test_generator.filenames
    submission['file'] = submission['file'].str.replace('\\', '')
    submission['file'] = submission['file'].str.replace('test', '')
    submission['species'] = class_list

    submission.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    main()
