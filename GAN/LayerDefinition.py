import AutoPackageInstaller as ap

ap.CheckAndInstall("tensorflow")

from tensorflow import keras
from tensorflow.keras import layers

class LayerDefinition():
    DiscriminatorInChannels = 0
    GeneratorInChannels = 0

    def __init__(self, discriminatorInChannels, generatorInChannels):
        self.DiscriminatorInChannels = discriminatorInChannels
        self.GeneratorInChannels = generatorInChannels

    def GetDiscriminator(self):
        return keras.Sequential(
            [
                keras.layers.InputLayer((28, 28, self.DiscriminatorInChannels)),
                layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same"),
                layers.LeakyReLU(alpha=0.2),
                layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same"),
                layers.LeakyReLU(alpha=0.2),
                layers.GlobalMaxPooling2D(),
                layers.Dense(1),
            ],
            name="discriminator",
        )

    def GetGenerator(self):
        return keras.Sequential(
            [
                keras.layers.InputLayer((self.GeneratorInChannels,)),
                # We want to generate 128 + num_classes coefficients to reshape into a
                # 7x7x(128 + num_classes) map.
                layers.Dense(7 * 7 * self.GeneratorInChannels),
                layers.LeakyReLU(alpha=0.2),
                layers.Reshape((7, 7, self.GeneratorInChannels)),
                layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
                layers.LeakyReLU(alpha=0.2),
                layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
                layers.LeakyReLU(alpha=0.2),
                layers.Conv2D(1, (7, 7), padding="same", activation="sigmoid"),
            ],
            name="generator",
        )