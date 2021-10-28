from DataLoader import DataLoader
from Classifier import Classifier
from Plotter import Plotter
from PlotData import PlotData
from tensorflow.keras.callbacks import History
from tensorflow import data
import tensorflow
import numpy



path = "C:\\Users\\Nana\\source\\Python\\Data\\Output"

data_loader = DataLoader()
fitting_data = data_loader.load_fitting_data(path)

classifier = Classifier()
epochs = 3





hist = classifier.train_model(train_ds, val_ds, num_classes, epochs)


'''graphdata = PlotData(epochs)

graphdata.add_dataset("Training Accuracy", hist.history['accuracy'])
graphdata.add_dataset("Validation Accuracy", hist.history['val_accuracy'])

#Plot some data
plotter = Plotter()
plotter.plot_graph(graphdata)
'''




    














