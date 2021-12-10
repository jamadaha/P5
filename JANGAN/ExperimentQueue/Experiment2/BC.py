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
            keras.layers.Conv2D(64, kernel_size=(4, 4), strides=(2, 2), padding="same", kernel_initializer=init)
        )
        model.add(
            keras.layers.LeakyReLU()
        )
        model.add(
            keras.layers.Conv2D(64, kernel_size=(4, 4), strides=(2, 2), padding="same", kernel_initializer=init)
        )
        model.add(
            keras.layers.LeakyReLU()
        )
        model.add(
            keras.layers.Flatten()
        )
        model.add(
            keras.layers.Dense(self.NumberOfClasses)
        )
        
        return model

Classifier.LayerDefinition.LayerDefinition = LayerDefinition