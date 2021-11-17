# Change functions and methods, to fit the goal of the experiment

from ProjectTools import AutoPackageInstaller as ap

ap.CheckAndInstall("tensorflow")
ap.CheckAndInstall("pydot")


import tensorflow as tf

import CGAN

class LayerDefinition(CGAN.LayerDefinition.LayerDefinition):
    def GetDiscriminator(self):
        disModel = tf.keras.Sequential(
            name='discriminator'
        )

        disModel.add(tf.keras.layers.InputLayer((28, 28, self.DiscriminatorInChannels)))
        disModel.add(
            tf.keras.layers.Conv2D(
                64, 
                (3, 3), 
                strides=(2, 2),
                padding='same')
        )
        disModel.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        disModel.add(
            tf.keras.layers.Conv2D(
                128, 
                (3, 3), 
                strides=(2, 2),
                padding='same')
        )
        disModel.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        disModel.add(tf.keras.layers.GlobalMaxPooling2D())
        disModel.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        
        print(disModel.summary())
        tf.keras.utils.plot_model(disModel, to_file='disModel.png', show_shapes=True)
        return disModel

CGAN.LayerDefinition.LayerDefinition = LayerDefinition