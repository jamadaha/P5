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
                tf.keras.layers.Flatten(input_shape=(28, 28)),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(self.NumberOfClasses)
            ],
            name="classifier",
        )
