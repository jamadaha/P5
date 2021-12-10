from ProjectTools import AutoPackageInstaller as ap

ap.CheckAndInstall("tensorflow")
ap.CheckAndInstall("pydot")

from tensorflow import keras

import Classifier
class LayerDefinition(Classifier.LayerDefinition.LayerDefinition):
    def GetClassifier(self):
        init = keras.initializers.RandomNormal(stddev=0.02)
        model = keras.Sequential(
            name='classifier'
        )
        model.add(
            keras.layers.InputLayer((self.ImageSize, self.ImageSize, self.ImageChannels))
        )
        model.add(
            keras.layers.Conv2D(128, kernel_size=3, strides=3, padding="same", activation="relu", kernel_initializer=init)
        )
        model.add(
            keras.layers.AveragePooling2D()
        )
        model.add(
            keras.layers.Conv2D(128, kernel_size=3, strides=3, padding="same", activation="relu", kernel_initializer=init)
        )
        model.add(
            keras.layers.AveragePooling2D()
        )
        model.add(
            keras.layers.Flatten()
        )
        model.add(
            keras.layers.Dense(self.NumberOfClasses)
        )
        
        return model

Classifier.LayerDefinition.LayerDefinition = LayerDefinition