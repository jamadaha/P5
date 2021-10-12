class Discriminator(object):
    """description of class"""

    def __init__(self, *args, **kwargs):
        self.model = _MakeDiscriminatorModel(self)
        return super().__init__(*args, **kwargs)

    def Train(labels, image):
        print("Disciminator Train")

    def Discriminator(labels, image):
        print("Disciminator Disciminate")

    def _MakeDiscriminatorModel(self):
        model = tf.keras.Sequential()
        model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
            input_shape=[IMAGE_RESOLUTION[0], IMAGE_RESOLUTION[1], 1]))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Flatten())
        model.add(layers.Dense(1))

        return model


