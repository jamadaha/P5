from ProjectTools import AutoPackageInstaller as ap

ap.CheckAndInstall("tensorflow")

import tensorflow as tf

class LayerDefinition():
    DiscriminatorInChannels = 0
    GeneratorInChannels = 0

    def __init__(self, discriminatorInChannels, generatorInChannels):
        self.DiscriminatorInChannels = discriminatorInChannels
        self.GeneratorInChannels = generatorInChannels


    # Good for calculating conv2d size change
    # https://madebyollin.github.io/convnet-calculator/
    def GetDiscriminator(self):
        model = tf.keras.Sequential(
            name='discriminator'
        )
        model.add(
            tf.keras.layers.InputLayer(
                (28, 28, self.DiscriminatorInChannels)
            )
        )
        model.add(
            tf.keras.layers.Conv2D(
                filters=64,
                kernel_size=(3, 3),
                strides=(2, 2),
                padding='same'
            )
        )
        model.add(
            tf.keras.layers.LeakyReLU(alpha=0.2)
        )
        model.add(
            tf.keras.layers.Conv2D(
                filters=128,
                kernel_size=(3, 3),
                strides=(2, 2),
                padding='same'
            )
        )
        model.add(
            tf.keras.layers.LeakyReLU(alpha=0.2)
        )
        model.add(
            tf.keras.layers.GlobalMaxPooling2D()
        )
        model.add(
            tf.keras.layers.Dense(1)
        )
        return model

    def GetGenerator(self):
        model = tf.keras.Sequential(
            name='generator'
        )
        model.add(
            tf.keras.layers.InputLayer(
                (self.GeneratorInChannels, )
            )
        )
        model.add(
            tf.keras.layers.Dense(
                7 * 7 * self.GeneratorInChannels
            )
        )
        model.add(
            tf.keras.layers.LeakyReLU(alpha=0.2)
        )
        model.add(
            tf.keras.layers.Reshape(
                (7, 7, self.GeneratorInChannels)
            )
        )
        model.add(
            tf.keras.layers.Conv2DTranspose(
                filters=128,
                kernel_size=(4, 4),
                strides=(2, 2),
                padding='same'
            )
        )
        model.add(
            tf.keras.layers.LeakyReLU(alpha=0.2)
        )
        model.add(
            tf.keras.layers.Conv2DTranspose(
                filters=128,
                kernel_size=(4, 4),
                strides=(2, 2),
                padding='same'
            )
        )
        model.add(
            tf.keras.layers.Conv2D(
                filters=1,
                kernel_size=(7, 7),
                strides=(1, 1),
                padding='same',
                activation='sigmoid'
            )
        )
        return model
