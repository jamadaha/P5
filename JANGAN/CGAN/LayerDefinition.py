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
        init = keras.initializers.RandomNormal(stddev=0.02)
        model = keras.Sequential(
            name='discriminator'
        )
        model.add(
            keras.layers.InputLayer((28, 28, self.DiscriminatorInChannels))
        )
        model = self.ConvLayer(
            model=model,
            init=init,
            activation=keras.layers.LeakyReLU(alpha=0.2),
            filterCount=64,
            kernelSize=3,
            stride=2,
            padding="same",
            batchNorm=False,
            dropout=False,
            dropAmount=0.2
        )
        model.add(
            keras.layers.AveragePooling2D()
        )
        model = self.ConvLayer(
            model=model,
            init=init,
            activation=keras.layers.LeakyReLU(alpha=0.2),
            filterCount=128,
            kernelSize=3,
            stride=2,
            padding="same",
            batchNorm=False,
            dropout=False
        )
        model.add(
            keras.layers.GlobalMaxPooling2D()
        )
        model.add(
            keras.layers.Dense(1)
        )
        return model

    def GetGenerator(self):
        init = keras.initializers.RandomNormal(stddev=0.02)
        model = keras.Sequential(
            name='generator'
        )
        model.add(
            keras.layers.InputLayer((self.GeneratorInChannels,))
        )
        model.add(
            keras.layers.Dense(7 * 7 * self.GeneratorInChannels, kernel_initializer=init)
        )
        model.add(
            keras.layers.LeakyReLU(alpha=0.2)
        )
        model.add(
            keras.layers.Reshape((7, 7, self.GeneratorInChannels))
        )
        model = self.ConvTransLayer(
            model=model,
            init=init,
            activation=keras.layers.LeakyReLU(alpha=0.2),
            filterCount=64,
            kernelSize=4,
            stride=2,
            padding="same",
            batchNorm=False,
            dropout=False,
            dropAmount=0.5
        )
        model = self.ConvTransLayer(
            model=model,
            init=init,
            activation=keras.layers.LeakyReLU(alpha=0.2),
            filterCount=64,
            kernelSize=4,
            stride=2,
            padding="same",
            batchNorm=False,
            dropout=False,
            dropAmount=0.5
        )
        model.add(
            keras.layers.Conv2D(
                1, 
                (7, 7), 
                padding="same",
                activation="sigmoid"
            )
        )
        return model

    
    def ConvLayer(self, model, init, activation, filterCount = 64, kernelSize = 4, stride = 2, padding="same", bias=True, batchNorm=True, dropout=True, dropAmount=0.5):
        model.add(
            keras.layers.Conv2D(
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
                keras.layers.BatchNormalization()
            )

        model.add(
            activation
        )

        if (dropout):
            model.add(
                keras.layers.Dropout(dropAmount)
            )

        return model

    def ConvTransLayer(self, model, init, activation, filterCount = 64, kernelSize = 4, stride = 2, padding="same", bias=True, batchNorm=True, dropout=True, dropAmount=0.5):
        model.add(
            keras.layers.Conv2DTranspose(
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
                keras.layers.BatchNormalization()
            )

        model.add(
            activation
        )

        if (dropout):
            model.add(
                keras.layers.Dropout(dropAmount)
            )

        return model
