from ProjectTools import AutoPackageInstaller as ap

ap.CheckAndInstall("tensorflow")

from tensorflow import keras

class LayerDefinition():
    DiscriminatorInChannels = 0
    GeneratorInChannels = 0
    ImageSize = 0

    def __init__(self, imageSize, discriminatorInChannels, generatorInChannels):
        self.DiscriminatorInChannels = discriminatorInChannels
        self.GeneratorInChannels = generatorInChannels
        self.ImageSize = imageSize

    def GetDiscriminator(self):
        return keras.Sequential(
            [
                keras.layers.InputLayer((self.ImageSize, self.ImageSize, self.DiscriminatorInChannels)),
                keras.layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same"),
                keras.layers.LeakyReLU(alpha=0.2),
                keras.layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same"),
                keras.layers.LeakyReLU(alpha=0.2),
                keras.layers.GlobalMaxPooling2D(),
                keras.layers.Dense(1),
            ],
            name="discriminator",
        )

    def GetGenerator(self):
        return keras.Sequential(
            [
                keras.layers.InputLayer((self.GeneratorInChannels,)),
                # We want to generate latent_dim + num_classes coefficients to reshape into a
                # 7x7x(latent_dim + num_classes) map.
                keras.layers.Dense(7 * 7 * self.GeneratorInChannels),
                keras.layers.LeakyReLU(alpha=0.2),
                keras.layers.Reshape((7, 7, self.GeneratorInChannels)),
                keras.layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
                keras.layers.LeakyReLU(alpha=0.2),
                keras.layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
                keras.layers.LeakyReLU(alpha=0.2),
                keras.layers.Conv2D(1, (7, 7), padding="same", activation="sigmoid"),
            ],
            name="generator",
        )