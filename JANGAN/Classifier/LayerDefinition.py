from ProjectTools import AutoPackageInstaller as ap

ap.CheckAndInstall("tensorflow")

import tensorflow as tf

class LayerDefinition():
    NumberOfClasses = 0

    def __init__(self, numberOfClasses):
        self.NumberOfClasses = numberOfClasses

    def GetClassifier(self):
        return tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer((28, 28, 1)),
                tf.keras.layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same"),
                tf.keras.layers.LeakyReLU(alpha=0.2),
                tf.keras.layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same"),
                tf.keras.layers.LeakyReLU(alpha=0.2),
                tf.keras.layers.GlobalMaxPooling2D(),
                tf.keras.layers.Dense(self.NumberOfClasses)
            ],
            name="classifier",
        )
