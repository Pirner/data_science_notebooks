import tensorflow as tf


class Trainer(object):
    def __init__(self, im_shape=(224, 224, 3)):
        self.im_shape = im_shape

    def train_model(self, train_gen, classes: int):
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=self.im_shape,
            include_top=False,
            weights='imagenet',
        )
        base_model.trainable = False

        inputs = tf.keras.Input(shape=self.im_shape)
        prediction_layer = tf.keras.layers.Dense(classes, activation='softmax')

        x = base_model(inputs, training=False)
        x = tf.keras.layers.Flatten()(x)

        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        # x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        outputs = prediction_layer(x)

        model = tf.keras.Model(inputs, outputs)
        model.summary()

        base_learning_rate = 0.0001
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                      loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                      metrics=['accuracy'])

        initial_epochs = 10

        # loss0, accuracy0 = model.evaluate(validation_dataset)

        # print("initial loss: {:.2f}".format(loss0))
        # print("initial accuracy: {:.2f}".format(accuracy0))

        history = model.fit(
            train_gen,
            epochs=initial_epochs,
            # validation_data=validation_dataset
        )
