from ProjectTools import AutoPackageInstaller as ap

ap.CheckAndInstall("tensorflow")
ap.CheckAndInstall("pydot")


import tensorflow as tf

import CGAN

class LayerDefinition(CGAN.LayerDefinition.LayerDefinition):
    def GetDiscriminator(self):
        disModel = tf.keras.Sequential(
            name="discriminator",
        )

        disModel.add(tf.keras.layers.InputLayer((28, 28, self.DiscriminatorInChannels)))
        disModel.add(
            tf.keras.layers.Conv2DTranspose(
                64, 
                kernel_size=(5, 5), 
                strides=(3, 3),
                padding='same',
                activation='relu'
            )
        )
        disModel.add(
            tf.keras.layers.Conv2D(
                32,
                kernel_size=(3, 3),
                strides=(4, 4),
                padding='same',
                activation='relu'
            )
        )
        disModel.add(
            tf.keras.layers.Conv2D(
                16,
                kernel_size=(5, 5),
                strides=(3, 3),
                padding='same',
                activation='relu'
            )
        )
        disModel.add(
            tf.keras.layers.Conv2D(
                1,
                kernel_size=(7, 7),
                strides=(7, 7),
                padding='same',
                activation='relu'
            )
        )
        disModel.add(
            tf.keras.layers.Flatten()
        )
        print(disModel.summary())
        return disModel

CGAN.LayerDefinition.LayerDefinition = LayerDefinition