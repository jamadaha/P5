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
        model.add(
            tf.keras.layers.LeakyReLU(alpha=0.2)
        )
        print(model.summary())
        tf.keras.utils.plot_model(model, to_file='disModel.png', show_shapes=True)
        return model

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