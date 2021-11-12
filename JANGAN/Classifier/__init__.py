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
from Classifier import DataLoader
from tensorflow.keras.models import load_model

ap.CheckAndInstall("tensorflow")

class Classifier():
    def __init__(self, epochs, retrain, modelname):
        self.save_dir =  Path( './training_models')
        self.model: Model

        #Metrics
        self.fit_history: tensorflow.keras.callbacks.History
        self.accuracy = []
        self.loss = []

        #Training variables
        self.epochs = epochs
        self.retrain = retrain
        self.modelname = modelname

    def TrainClassifier(self, data_path: str):
        #Sets up the model for training
        self.model = self._CreateModel_(LayerConfigObject.LayerConfigObject(), CompilerConfigObject.CompilerConfigObject())
        
        #Mount data from GAN
        dataLoader = DataLoader.DataLoader()
        data = dataLoader.load_fitting_data(data_path)
        
        #Train the model with the mounted data
        self.model = self._TrainModelCallback_(self.model, data, self.epochs, self.retrain, self.modelname)
        return self.model

    def _EvaluateOnRealData_(self, data_path: str, validation_split = 0.2, subset = "validaton", seed = 123):
        dataLoader = DataLoader.DataLoader()
        data = dataLoader.load_data_set(data_path, validation_split, subset, seed)
        score = self.model.evaluate(x = data, verbose = 1)
        self.loss = score[0]
        self.accuracy = score[1]

    def ProduceStatistics(self, input_path: str, validation_split = 0.2, subset = "validation", seed = 123):
        self._EvaluateOnRealData_(input_path, validation_split, subset, seed)
        return self.accuracy
    
    def _CreateModel_(self, layers: LayerConfigObject.LayerConfigObject, compile_config: CompilerConfigObject.CompilerConfigObject):
        model = LetterModel.LetterModel(layers)
        model.compile(compile_config)
        return model.sequential

    def _TestModel_(self, model: Model, test_data: tensorflow.data.Dataset):
        return model.evaluate(test_data)

    def _TrainModelCallback_(self, model: Model, fd: FitData.FitData, epochs: int, retrain = False, model_name = "my_model"):
        cm = model
        save_path = self.save_dir / (model_name + '.h5')
        self.make_dir(self.save_dir)

        if(retrain):
            self.fit_history = cm.fit(fd.get_train_data(), epochs=epochs)
            cm.save(save_path)
            return cm

        if(not (save_path.exists())):
            self.fit_history = cm.fit(fd.get_train_data(), epochs=epochs)
            cm.save(save_path)
            return cm

        else:
            cm = load_model(save_path)
            return cm

    def _PredictData_(self, model: Model, data: tensorflow.data.Dataset):
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


