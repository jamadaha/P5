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
                filters=64, 
                kernel_size=7, 
                strides=3,
                padding="valid"
            )
        )
        disModel.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        disModel.add(
            tf.keras.layers.Conv2D(
                filters=32, 
                kernel_size=2, 
                strides=2,
                padding="valid"
            )
        )
        disModel.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        disModel.add(tf.keras.layers.Flatten())
        disModel.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        print(disModel.summary())
        tf.keras.utils.plot_model(disModel, to_file='disModel.png', show_shapes=True)
        return disModel

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
        #genModel.add(tf.keras.layers.BatchNormalization())
        genModel.add(tf.keras.layers.LeakyReLU(alpha=0.2))
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
        genModel.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        genModel.add(
            tf.keras.layers.Conv2DTranspose(
                self.GeneratorInChannels, 
                kernel_size=(4, 4),
                strides=(2, 2),
                padding='same'
            )
        )
        genModel.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        genModel.add(
            tf.keras.layers.Conv2D(
                1,
                kernel_size=(7, 7),
                padding='same',
                activation='sigmoid'
            )
        )

        
        print(genModel.summary())
        tf.keras.utils.plot_model(genModel, to_file='genModel.png', show_shapes=True)
        return genModel

CGAN.LayerDefinition.LayerDefinition = LayerDefinition