from DataLoader.DataLoader import DataLoader
from DataLoader.FitData import FitData
from Classifier.Classifier import Classifier
from Classifier.Classifier import LayerConfigObject
from Classifier.Classifier import CompilerConfigObject
from Plotter.Plotter import Plotter
from Plotter.PlotData import PlotData
from tensorflow.keras.callbacks import History
from tensorflow import data
import tensorflow
import numpy
import os



data_path = "..\\..\\Data\\Output"


def run_training(epochs: int, path: str, layers = LayerConfigObject(), config = CompilerConfigObject(), retrain = False, model_name = "my_model"):
    fitting_data = load_data(path) 
    classifier = Classifier()
    layers.add_dense_layer(fitting_data.num_classes)
    model = classifier.create_model(layers, config)
    return classifier.train_model_callback(model, fitting_data, epochs, retrain, model_name)

def load_data(path: str):
    data_loader = DataLoader()
    fitting_data = data_loader.load_fitting_data(path)
    return fitting_data


run_training(3, path = data_path, retrain = False, model_name="my_model")








    














