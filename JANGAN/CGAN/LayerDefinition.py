from ProjectTools import AutoPackageInstaller as ap

ap.CheckAndInstall("tensorflow")

import tensorflow as tf

class LayerDefinition():
    DiscriminatorInChannels = 0
    GeneratorInChannels = 0
    ImageSize = 0

    def __init__(self, imageSize, discriminatorInChannels, generatorInChannels):
        self.DiscriminatorInChannels = discriminatorInChannels
        self.GeneratorInChannels = generatorInChannels
        self.ImageSize = imageSize

    def GetDiscriminator(self):
        return tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer((self.ImageSize, self.ImageSize, self.DiscriminatorInChannels)),
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
                # We want to generate latent_dim + num_classes coefficients to reshape into a
                # 7x7x(latent_dim + num_classes) map.
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