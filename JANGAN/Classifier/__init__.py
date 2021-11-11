from ProjectTools import AutoPackageInstaller as ap
import os
from pathlib import Path
import numpy
import tensorflow
from tensorflow.keras.callbacks import ModelCheckpoint
from multipledispatch import dispatch
from tensorflow.keras import Model
from Classifier import FitData
from Classifier import LayerConfigObject
from Classifier import CompilerConfigObject
from Classifier import LetterModel
from tensorflow.keras.models import load_model

ap.CheckAndInstall("tensorflow")

class Classifier():
    def __init__(self):
        pass

    def TrainClassifier(self):
        # Train the classifier, or use a checkpoinht if its there
        pass

    def ProduceStatistics(self):
        # Output from the classifier. Could be accuracy or something
        pass
