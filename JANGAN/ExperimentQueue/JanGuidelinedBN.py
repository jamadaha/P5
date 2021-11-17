# This is an extension of JanGuidelined
# Now also uses batch norm

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
                padding='same'
            )
        )
        disModel.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        disModel.add(
            tf.keras.layers.Conv2D(
                32,
                kernel_size=(3, 3),
                strides=(4, 4),
                padding='same'
            )
        )
        disModel.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        disModel.add(tf.keras.layers.BatchNormalization())
        disModel.add(
            tf.keras.layers.Conv2D(
                16,
                kernel_size=(5, 5),
                strides=(3, 3),
                padding='same'
            )
        )
        disModel.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        disModel.add(tf.keras.layers.BatchNormalization())
        disModel.add(
            tf.keras.layers.Conv2D(
                1,
                kernel_size=(7, 7),
                strides=(7, 7),
                padding='same'
            )
        )
        disModel.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        disModel.add(tf.keras.layers.BatchNormalization())
        disModel.add(
            tf.keras.layers.Flatten()
        )
        print(disModel.summary())
        return disModel

    def GetGenerator(self):
        genModel = tf.keras.Sequential(
            name='generator'
        )

        genModel.add(tf.keras.layers.InputLayer(self.GeneratorInChannels,))
        genModel.add(
            tf.keras.layers.Dense(
                7 * 7 * self.GeneratorInChannels, 
                activation='relu'
                )
        )
        genModel.add(tf.keras.layers.BatchNormalization())
        genModel.add(
            tf.keras.layers.Reshape(
                (7, 7, self.GeneratorInChannels)
            )
        )
        genModel.add(
            tf.keras.layers.Conv2DTranspose(
                self.GeneratorInChannels, 
                kernel_size=(5, 5),
                strides=(2, 2),
                padding='same',
                activation='relu'
            )
        )
        genModel.add(tf.keras.layers.BatchNormalization())
        genModel.add(
            tf.keras.layers.Conv2DTranspose(
                self.GeneratorInChannels, 
                kernel_size=(5, 5),
                strides=(2, 2),
                padding='same',
                activation='relu'
            )
        )
        genModel.add(tf.keras.layers.BatchNormalization())
        genModel.add(
            tf.keras.layers.Conv2D(
                1,
                kernel_size=(7, 7),
                padding='same',
                activation='tanh'
            )
        )

        
        print(genModel.summary())
        tf.keras.utils.plot_model(genModel, to_file='genModel.png', show_shapes=True)
        return genModel

CGAN.LayerDefinition.LayerDefinition = LayerDefinition