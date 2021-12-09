from ProjectTools import AutoPackageInstaller as ap

ap.CheckAndInstall("tensorflow")

import tensorflow as tf

class LayerDefinition():
    NumberOfClasses = 0
    ImageSize = 0
    ImageChannels = 0

    def __init__(self, numberOfClasses, imageSize, imageChannels):
        self.NumberOfClasses = numberOfClasses
        self.ImageSize = imageSize
        self.ImageChannels = imageChannels

    def GetClassifier(self):
        return tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer((self.ImageSize, self.ImageSize, self.ImageChannels)),
                tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu'),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(self.NumberOfClasses)
            ],
            name="classifier",
        )
