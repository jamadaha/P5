from DataLoader import DataLoader
from Classifier import Classifier
from Plotter import Plotter
from tensorflow import data
from tensorflow import History
import tensorflow
import numpy

path = "C:\\Users\\Nana\\source\\Python\\Data\\Output"

data_loader = DataLoader()
data_loader.load_from_dir(path)

train_ds = data_loader.get_training_ds()
val_ds = data_loader.get_validation_ds()

num_classes = len(train_ds.class_names)

train_ds = data_loader.preprocess_images(train_ds)
val_ds = data_loader.preprocess_images(val_ds)

classifier = Classifier()

hist = classifier.train_model(train_ds, val_ds, num_classes, 3)







plotter = Plotter()



    














