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
        model = self.ConvLayer(
            model=model,
            init=init,
            activation=tf.keras.layers.LeakyReLU(alpha=0.2),
            filterCount=64,
            kernelSize=3,
            stride=2,
            padding="same",
            batchNorm=True,
            dropout=False
        )
        model = self.ConvLayer(
            model=model,
            init=init,
            activation=tf.keras.layers.LeakyReLU(alpha=0.2),
            filterCount=128,
            kernelSize=3,
            stride=2,
            padding="same",
            batchNorm=True,
            dropout=False
        )
        model.add(
            tf.keras.layers.GlobalMaxPooling2D()
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
            tf.keras.layers.LeakyReLU(alpha=0.2)
        )
        model.add(
            tf.keras.layers.Reshape((7, 7, self.GeneratorInChannels))
        )
        model = self.ConvTransLayer(
            model=model,
            init=init,
            activation=tf.keras.layers.LeakyReLU(alpha=0.2),
            filterCount=128,
            kernelSize=4,
            stride=2,
            padding="same",
            batchNorm=True,
            dropout=False,
            dropAmount=0.5
        )
        model = self.ConvTransLayer(
            model=model,
            init=init,
            activation=tf.keras.layers.LeakyReLU(alpha=0.2),
            filterCount=128,
            kernelSize=4,
            stride=2,
            padding="same",
            batchNorm=True,
            dropout=False,
            dropAmount=0.5
        )
        model.add(
            tf.keras.layers.Conv2D(
                1, 
                (7, 7), 
                padding="same",
                activation="sigmoid"
            )
        )
        return model

CGAN.LayerDefinition.LayerDefinition = LayerDefinition

class CGANMLModel(CGAN.CGANMLModel.CGANMLModel):
    def GetOptimizer(self):
        (disSchedule, genSchedule) = self.GetLearningSchedule()
        return (
            tf.keras.optimizers.Adam(learning_rate=disSchedule),
            tf.keras.optimizers.Adam(learning_rate=genSchedule)
        )
    
    def GetLossFunction(self): 
        return tf.keras.losses.BinaryCrossentropy(from_logits=True)


CGAN.CGANMLModel.CGANMLModel = CGANMLModel
