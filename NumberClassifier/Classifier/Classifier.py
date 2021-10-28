import os
from pathlib import Path
import numpy
import tensorflow
from tensorflow.keras.callbacks import ModelCheckpoint
from multipledispatch import dispatch
from tensorflow.keras import Model
from DataLoader.FittingData import FittingData
from Classifier.LayerConfigObject import LayerConfigObject
from Classifier.CompilerConfigObject import CompilerConfigObject
from Classifier.LetterModel import LetterModel



class Classifier(object):
    """description of class"""
    
    def __init__(self):
        self.training_data: tensorflow.data.Dataset
        self.test_data: tensorflow.data.Dataset


        #Variables for saving models
        self.save_dir =  Path( './training_models')
        self.save_path = self.save_dir / 'cp.ckpt'
        

    @dispatch(list, str)
    def set_model(self, layers: list, name: str):
        self.model = tensorflow.keras.Sequential(layers, name)

    @dispatch(tensorflow.keras.Model)
    def set_model(self, model: tensorflow.keras.Model):
       self.model = model

    def mount_data(training_data, test_data):
        self.training_data: training_data
        self.test_data: test_data

    def create_model(self, layers: LayerConfigObject, compile_config: CompilerConfigObject):
        model = LetterModel(layers)
        model.compile(compile_config)
        return model

    def train_model(self, data: FittingData, model: Model, ep: int):
        history =  model.fit(
                    train_data,
                    validation_data=val_data,
                    epochs=ep
                    )

        return history

    def train_model_with_callback(self, data: FittingData, epochs: int, retrain: bool):

        if(retrain == True):
            if(not self.save_dir.exists()):
                self.save_dir.mkdir(parents=True, exist_ok=True)

            model_callback = ModelCheckpoint(filepath = self.save_path, save_weights_only = False, verbose=1)

            history =  model.fit(
                        train_data,
                        validation_data=val_data,
                        epochs=ep, callbacks=[model_callback]
                        )
        return history




    def test_model(self, test_data: tensorflow.data.Dataset):
        return self.model.evaluate(test_data)

    def load_model_from_dir(self, path: str):
        
        
        
        
        
        
        


      