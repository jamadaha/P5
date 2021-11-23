from ProjectTools import AutoPackageInstaller as ap

ap.CheckAndInstall("tensorflow")
ap.CheckAndInstall("pydot")


import tensorflow as tf

import CGAN

class LayerDefinition(CGAN.LayerDefinition.LayerDefinition):
    def GetGenerator(self):
        genModel = tf.keras.Sequential(
            name='generator'
        )

        genModel.add(tf.keras.layers.InputLayer(self.GeneratorInChannels,))
        genModel.add(
            tf.keras.layers.Dense(
                7 * 7 * self.GeneratorInChannels
                )
        )
        genModel.add(tf.keras.layers.LeakyReLU())
        genModel.add(
            tf.keras.layers.Reshape(
                (7, 7, self.GeneratorInChannels)
            )
        )
        genModel.add(
            tf.keras.layers.Conv2DTranspose(
                self.GeneratorInChannels, 
                kernel_size=(4, 4),
                strides=(2, 2),
                padding='same'
            )
        )
        genModel.add(tf.keras.layers.LeakyReLU())
        genModel.add(
            tf.keras.layers.Conv2DTranspose(
                self.GeneratorInChannels, 
                kernel_size=(4, 4),
                strides=(2, 2),
                padding='same'
            )
        )
        genModel.add(tf.keras.layers.LeakyReLU())
        genModel.add(
            tf.keras.layers.Conv2D(
                1,
                kernel_size=(3, 3),
                padding='same',
                activation='tanh'
            )
        )

        
        print(genModel.summary())
        tf.keras.utils.plot_model(genModel, to_file='genModel.png', show_shapes=True)
        return genModel

CGAN.LayerDefinition.LayerDefinition = LayerDefinition