from ProjectTools import AutoPackageInstaller as ap

ap.CheckAndInstall("tensorflow")
ap.CheckAndInstall("pathlib")
ap.CheckAndInstall("tensorflow")
ap.CheckAndInstall("multipledispatch")
ap.CheckAndInstall("tensorflow")

import os
from pathlib import Path
import numpy
import tensorflow
from tensorflow.keras.callbacks import ModelCheckpoint
from multipledispatch import dispatch
from tensorflow.keras import Model
from Classifier import LayerConfigObject
from Classifier import CompilerConfigObject
from Classifier import LetterModel
from Classifier import DataLoader
from tensorflow.keras.models import load_model
from DatasetLoader import DatasetLoader
from DatasetLoader.DatasetFormatter import BulkDatasetFormatter
from DatasetLoader.FitData import FitData

class Classifier():
    #Model
    save_dir: Path
    model: Model
    #Metrics
    fit_history: tensorflow.keras.callbacks.History
    accuracy: []
    loss: []
    #Training variables
    epochs: int
    retrain: bool
    modelname: str
    #Data and data format variables
    TrainDir = ""
    TestDir = ""
    BatchSize = -1
    ImageHeight = -1
    ImageWidth = -1
    Seed = -1
    Split = -0.0
    FitData = []
    

    def __init__(self, epochs, retrain, modelname, model_path, traindir, testdir, batchsize, imageheight, imagewidth, seed, split):
        self.save_dir =  Path(model_path)
        self.accuracy = []
        self.loss = []
        self.epochs = epochs
        self.retrain = retrain
        self.modelname = modelname
        self.BatchSize = batchsize
        self.ImageHeight = imageheight
        self.ImageWidth = imagewidth
        self.Seed = seed
        self.TrainDir = traindir
        self.TestDir = testdir
        self.Split = split
        
    def LoadData(self):
        dl = DatasetLoader(self.TrainDir, self.TestDir, (self.ImageHeight, self.ImageWidth))
        fitData = dl.LoadFittingData(self.TrainDir, batch_size=self.BatchSize, img_height = self.ImageHeight, img_width=self.ImageWidth, seed=self.Seed, split=self.Split)
        self.FitData.append(fitData)



    def TrainClassifier(self):
        data = self.FitData.pop()
        layers = LayerConfigObject.LayerConfigObject()
        layers.AddDenseLayer(data.num_classes)
        #Sets up the model for training
        self.model = self.__CreateModel(layers, CompilerConfigObject.CompilerConfigObject())
        #Train the model with the mounted data
        self.model = self.__TrainModelCallback_(self.model, data, self.epochs, self.retrain, self.modelname)
        return self.model

    def __EvaluateOnData(self, data: tensorflow.data.Dataset , validation_split = 0.2, subset = "validaton", seed = 123):
        score = self.model.evaluate(x = data, verbose = 1)
        self.loss = score[0]
        self.accuracy = score[1]

    def ProduceStatistics(self, input_path: str, validation_split = 0.2, subset = "validation", seed = 123):
        self.__EvaluateOnData(input_path, validation_split, subset, seed)
        return self.accuracy
    
    def __CreateModel(self, layers: LayerConfigObject.LayerConfigObject, compile_config: CompilerConfigObject.CompilerConfigObject):
        model = LetterModel.LetterModel(layers)
        model.compile(compile_config)
        return model.sequential

    def __TestModel(self, model: Model, test_data: tensorflow.data.Dataset):
        return model.evaluate(test_data)

    def __TrainModelCallback_(self, model: Model, fd: FitData, epochs: int, retrain = False, model_name = "my_model"):
        cm = model
        save_path = self.save_dir / (model_name + '.h5')
        self.save_dir.mkdir(parents = True, exist_ok = True)

        if(retrain):
            self.fit_history = cm.fit(fd.train_data, validation_data=fd.val_data, epochs=epochs)
            cm.save(save_path)
            return cm

        if(not save_path.exists()):
            self.fit_history = cm.fit(fd.train_data, validation_data=fd.val_data, epochs=epochs)
            cm.save(save_path)
            return cm

        else:
            cm = load_model(save_path)
            return cm

    def __PredictData(self, model: Model, data: tensorflow.data.Dataset):
        try: 
            return model.predict(data)
        except:
            raise PredictedOnUntrainedModelException("Tried to predict over untrained model.")

    class PredictedOnUntrainedModelException(Exception):
        pass


