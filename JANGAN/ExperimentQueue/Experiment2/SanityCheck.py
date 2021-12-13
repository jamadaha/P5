from ProjectTools import AutoPackageInstaller as ap

ap.CheckAndInstall("tensorflow")
ap.CheckAndInstall("pydot")

import tensorflow as tf

import CGAN
class LayerDefinition(CGAN.LayerDefinition.LayerDefinition):
    def GetDiscriminator(self):
        model = tf.keras.Sequential(
            name='discriminator'
        )
        model.add(
            tf.keras.layers.InputLayer((28, 28, self.DiscriminatorInChannels))
        )
        model.add(
            tf.keras.layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same")
        )
        model.add(
            tf.keras.layers.LeakyReLU(alpha=0.2)
        )
        model.add(
            tf.keras.layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same")
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
            tf.keras.layers.InputLayer((self.GeneratorInChannels,))
        )
        model.add(
            tf.keras.layers.Dense(7 * 7 * self.GeneratorInChannels)
        )
        model.add(
            tf.keras.layers.LeakyReLU(alpha=0.2)
        )
        model.add(
            tf.keras.layers.Reshape((7, 7, self.GeneratorInChannels))
        )
        model.add(
            tf.keras.layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same")
        )
        model.add(
            tf.keras.layers.LeakyReLU(alpha=0.2)
        )
        model.add(
            tf.keras.layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same")
        )
        model.add(
            tf.keras.layers.LeakyReLU(alpha=0.2)
        )
        model.add(
            tf.keras.layers.Conv2D(1, (7, 7), padding="same", activation="sigmoid")
        )
        return model

CGAN.LayerDefinition.LayerDefinition = LayerDefinition