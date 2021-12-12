from ProjectTools import AutoPackageInstaller as ap

ap.CheckAndInstall("tensorflow")

from tensorflow import keras

class LayerDefinition():
    NumberOfClasses = 0
    ImageSize = 0
    ImageChannels = 0

    def __init__(self, numberOfClasses, imageSize, imageChannels):
        self.NumberOfClasses = numberOfClasses
        self.ImageSize = imageSize
        self.ImageChannels = imageChannels

    def GetClassifier(self):
        return keras.Sequential(
            [
                keras.layers.InputLayer((self.ImageSize, self.ImageSize, self.ImageChannels)),
                keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu'),
                keras.layers.MaxPooling2D(),
                keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
                keras.layers.MaxPooling2D(),
                keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
                keras.layers.MaxPooling2D(),
                keras.layers.Flatten(),
                keras.layers.Dense(128, activation='relu'),
                keras.layers.Dense(self.NumberOfClasses)
            ],
            name="classifier",
        )
