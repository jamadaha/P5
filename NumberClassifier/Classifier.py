import DataLoader
import tensorflow
import numpy
from multipledispatch import dispatch


class Classifier(object):
    """description of class"""
    
    def __init__(self):
        self.model = tensorflow.keras.Sequential(layers = [tensorflow.keras.layers.Flatten(input_shape=(28,28)), 
                                                  tensorflow.keras.layers.Dense(128, activation='relu'), 
                                                  tensorflow.keras.layers.Dense(10)], name = "simpleModel")
        self.training_data: tensorflow.data.Dataset
        self.test_data: tensorflow.data.Dataset

        self.optimizer = "adam"
        self.loss = tensorflow.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.metrics = ['accuracy']

    @dispatch(list, str)
    def set_model(self, layers: list, name):
        self.model = tensorflow.keras.Sequential(layers, name)

    @dispatch(tensorflow.keras.Model)
    def set_model(self, model: tensorflow.keras.Model):
       self.model = model

    def mount_data(training_data, test_data):
        self.training_data: training_data
        self.test_data: test_data

    @dispatch(tensorflow.data.Dataset, int)
    def train_model(self, training_data: tensorflow.data.Dataset, epoch: int):
        self.model.compile(training_data)

    @dispatch(tensorflow.data.Dataset, tensorflow.data.Dataset, int, int)
    def train_model(self, train_data: tensorflow.data.Dataset, val_data: tensorflow.data.Dataset, num_classes: int, ep: int):
        
        model = tensorflow.keras.Sequential([
                    tensorflow.keras.layers.Rescaling(1./255),
                    tensorflow.keras.layers.Conv2D(32, 3, activation='relu'),
                    tensorflow.keras.layers.MaxPooling2D(),
                    tensorflow.keras.layers.Conv2D(32, 3, activation='relu'),
                    tensorflow.keras.layers.MaxPooling2D(),
                    tensorflow.keras.layers.Conv2D(32, 3, activation='relu'),
                    tensorflow.keras.layers.MaxPooling2D(),
                    tensorflow.keras.layers.Flatten(),
                    tensorflow.keras.layers.Dense(128, activation='relu'),
                    tensorflow.keras.layers.Dense(num_classes)
            ])

        model.compile( 
                    optimizer='adam',
                    loss=tensorflow.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])

        return model.fit(
                    train_data,
                    validation_data=val_data,
                    epochs=ep
                )

    def test_model(self, test_data: tensorflow.data.Dataset):
        return self.model.evaluate(test_data)
        
        
        
        
        
        
        


      