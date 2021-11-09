import os
from pathlib import Path
import numpy
import tensorflow
from tensorflow.keras.callbacks import ModelCheckpoint
from multipledispatch import dispatch
from tensorflow.keras import Model
from DataLoader.FitData import FitData
from Classifier.LayerConfigObject import LayerConfigObject
from Classifier.CompilerConfigObject import CompilerConfigObject
from Classifier.LetterModel import LetterModel
from tensorflow.keras.models import load_model
import PathUtil



class Classifier(object):
    """A class for handling model creation, compilation, training and evaluation. """

    def __init__(self):
        self.save_dir =  Path( './training_models')
    
    def create_model(self, layers: LayerConfigObject, compile_config: CompilerConfigObject):
        model = LetterModel(layers)
        model.compile(compile_config)
        return model.sequential

    def test_model(self, model: Model, test_data: tensorflow.data.Dataset):
        return model.evaluate(test_data)

    def train_model_callback(self, model: Model, fd: FitData, epochs: int, retrain = False, model_name = "my_model"):
        cm = model
        save_path = self.save_dir / (model_name + '.h5')
        self.make_dir(self.save_dir)

        if(retrain):
            cm.fit(fd.get_train_data(), epochs=epochs)
            cm.save(save_path)
            return cm

        if(not (save_path.exists())):
            cm.fit(fd.get_train_data(), epochs=epochs)
            cm.save(save_path)
            return cm

        else:
            cm = load_model(save_path)
            return cm

    def prepare_model(self, layers: LayerConfigObject, compile_config: CompilerConfigObject, fit_data: FitData, epochs: int, model_name = "my_model", retrain = False):
        model = self.create_model(layers, compile_config)
        return self.train_model_callback(model, fit_data, epochs, retrain, model_name)

    def predict_data(self, model: Model, data: tensorflow.data.Dataset):
        try: 
            return model.predict(data)
        except:
            print("Exception thrown. Tried to predict over untrained model.")

    #Utility functions for handling paths.

    def make_dir(self, path: Path):
        if(path.exists()):
            return path
        else:
            path.mkdir(parents=True, exist_ok=True)



        
        






    

        
        
        
        


      