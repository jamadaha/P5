from ProjectTools import AutoPackageInstaller as ap

ap.CheckAndInstall("tensorflow")

import tensorflow as tf

class LayerDefinition():
    DiscriminatorInChannels = 0
    GeneratorInChannels = 0

    def __init__(self, discriminatorInChannels, generatorInChannels):
        self.DiscriminatorInChannels = discriminatorInChannels
        self.GeneratorInChannels = generatorInChannels

    def GetDiscriminator(self):
        return tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer((28, 28, self.DiscriminatorInChannels)),
                tf.keras.layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same"),
                tf.keras.layers.LeakyReLU(alpha=0.2),
                tf.keras.layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same"),
                tf.keras.layers.LeakyReLU(alpha=0.2),
                tf.keras.layers.GlobalMaxPooling2D(),
                tf.keras.layers.Dense(1),
            ],
            name="discriminator",
        )

    def GetGenerator(self):
        return tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer((self.GeneratorInChannels,)),
                # We want to generate 128 + num_classes coefficients to reshape into a
                # 7x7x(128 + num_classes) map.
                tf.keras.layers.Dense(7 * 7 * self.GeneratorInChannels),
                tf.keras.layers.LeakyReLU(alpha=0.2),
                tf.keras.layers.Reshape((7, 7, self.GeneratorInChannels)),
                tf.keras.layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
                tf.keras.layers.LeakyReLU(alpha=0.2),
                tf.keras.layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
                tf.keras.layers.LeakyReLU(alpha=0.2),
                tf.keras.layers.Conv2D(1, (7, 7), padding="same", activation="sigmoid"),
            ],
            name="generator",
        )
    
    def ConvLayer(self, model, init, activation, filterCount = 64, kernelSize = 4, stride = 2, padding="same", bias=True, batchNorm=True, dropout=True, dropAmount=0.5):
        model.add(
            tf.keras.layers.Conv2D(
            filters=filterCount,
            kernel_size=kernelSize,
            strides=stride,
            padding=padding,
            use_bias=bias,
            kernel_initializer=init,
            bias_initializer=init
            )
        )

        if (batchNorm):
            model.add(
                tf.keras.layers.BatchNormalization()
            )

        model.add(
            activation
        )

        if (dropout):
            model.add(
                tf.keras.layers.Dropout(dropAmount)
            )

        return model

    def ConvTransLayer(self, model, init, activation, filterCount = 64, kernelSize = 4, stride = 2, padding="same", bias=True, batchNorm=True, dropout=True, dropAmount=0.5):
        model.add(
            tf.keras.layers.Conv2DTranspose(
            filters=filterCount,
            kernel_size=kernelSize,
            strides=stride,
            padding=padding,
            use_bias=bias,
            kernel_initializer=init,
            bias_initializer=init
            )
        )

        if (batchNorm):
            model.add(
                tf.keras.layers.BatchNormalization()
            )

        model.add(
            activation
        )

        if (dropout):
            model.add(
                tf.keras.layers.Dropout(dropAmount)
            )

        return model
