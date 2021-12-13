from ProjectTools import AutoPackageInstaller as ap

ap.CheckAndInstall("tensorflow")
ap.CheckAndInstall("pydot")

import tensorflow as tf

import CGAN
class LayerDefinition(CGAN.LayerDefinition.LayerDefinition):
    def GetDiscriminator(self):
        init = tf.keras.initializers.RandomNormal(stddev=0.02)
        model = tf.keras.Sequential(
            name='discriminator'
        )
        model.add(
            tf.keras.layers.InputLayer((28, 28, self.DiscriminatorInChannels))
        )
        model.add(
            tf.keras.layers.Conv2D(64, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init)
        )
        model.add(
            tf.keras.layers.BatchNormalization()
        )
        model.add(
            tf.keras.layers.LeakyReLU(alpha=0.2)
        )
        model.add(
            tf.keras.layers.Conv2D(128, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init)
        )
        model.add(
            tf.keras.layers.BatchNormalization()
        )
        model.add(
            tf.keras.layers.LeakyReLU(alpha=0.2)
        )
        model.add(
            tf.keras.layers.Conv2D(128, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init)
        )
        model.add(
            tf.keras.layers.BatchNormalization()
        )
        model.add(
            tf.keras.layers.LeakyReLU(alpha=0.2)
        )
        model.add(
            tf.keras.layers.Flatten()
        )
        model.add(
            tf.keras.layers.Dense(1)
        )
        return model

    def GetGenerator(self):
        init = tf.keras.initializers.RandomNormal(stddev=0.02)
        model = tf.keras.Sequential(
            name='generator'
        )
        model.add(
            tf.keras.layers.InputLayer((self.GeneratorInChannels,))
        )
        model.add(
            tf.keras.layers.Dense(7 * 7 * self.GeneratorInChannels, kernel_initializer=init)
        )
        model.add(
            tf.keras.layers.BatchNormalization()
        )
        model.add(
            tf.keras.layers.LeakyReLU(alpha=0.2)
        )
        model.add(
            tf.keras.layers.Reshape((7, 7, self.GeneratorInChannels))
        )
        model.add(
            tf.keras.layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init)
        )
        model.add(
            tf.keras.layers.BatchNormalization()
        )
        model.add(
            tf.keras.layers.LeakyReLU(alpha=0.2)
        )
        model.add(
            tf.keras.layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init)
        )
        model.add(
            tf.keras.layers.BatchNormalization()
        )
        model.add(
            tf.keras.layers.LeakyReLU(alpha=0.2)
        )
        model.add(
            tf.keras.layers.Conv2D(1, (7, 7), padding="same", activation="sigmoid", kernel_initializer=init)
        )
        return model

CGAN.LayerDefinition.LayerDefinition = LayerDefinition

class CGANMLModel(CGAN.CGANMLModel.CGANMLModel):
    def GetOptimizer(self):
        (disSchedule, genSchedule) = self.__GetLearningSchedule()
        return (
            tf.keras.optimizers.Adam(learning_rate=disSchedule, beta_1=0.5),
            tf.keras.optimizers.Adam(learning_rate=genSchedule, beta_1=0.5)
        )


CGAN.CGANMLModel.CGANMLModel = CGANMLModel
