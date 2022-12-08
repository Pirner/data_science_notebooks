import tensorflow as tf


class ModelCreator(object):
    def __init__(self):
        pass

    @staticmethod
    def create_base_model(image_size):
        """
        create the base model
        :return:
        """
        model = tf.keras.models.Sequential()

        # Input layer
        # Can be omitted, you can specify the input_shape in other layers
        model.add(tf.keras.layers.InputLayer(input_shape=(image_size, image_size, 3,)))

        # Here we add a 2D Convolution layer
        # Check https://keras.io/api/layers/convolution_layers/convolution2d/ for more info
        model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))

        # Max Pool layer
        # It down samples the input representation within the pool_size size
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

        # Normalization layer
        # The layer normalizes its output using the mean and standard deviation of the current batch of inputs.
        model.add(tf.keras.layers.BatchNormalization())

        # 2D Convolution layer
        model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu'))

        # Max Pool layer
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

        # Normalization layer
        model.add(tf.keras.layers.BatchNormalization())

        # 2D Convolution layer
        model.add(tf.keras.layers.Conv2D(128, kernel_size=(3, 3), strides=(1, 1), activation='relu'))

        # Max Pool layer
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

        # Normalization layer
        model.add(tf.keras.layers.BatchNormalization())

        # 2D Convolution layer
        model.add(tf.keras.layers.Conv2D(128, kernel_size=(3, 3), strides=(1, 1), activation='relu'))

        # Max Pool layer
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

        # Global Max Pool layer
        model.add(tf.keras.layers.GlobalMaxPool2D())

        # Dense Layers after flattening the data
        model.add(tf.keras.layers.Flatten())

        model.add(tf.keras.layers.Dense(128, activation='relu'))

        # Dropout
        # is used to nullify the outputs that are very close to zero and thus can cause over fitting.
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(64, activation='relu'))

        # Normalization layer
        model.add(tf.keras.layers.BatchNormalization())

        # Add Output Layer
        model.add(tf.keras.layers.Dense(12, activation='softmax'))  # = 12 predicted classes

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()
        return model
