from ProjectTools import AutoPackageInstaller as ap

import os

from DatasetLoader import DatasetLoader as dl
from DatasetLoader import DatasetFormatter as df
from ProjectTools import BaseKerasModelTrainer as bk

class BaseMLModel():
    BatchSize = -1
    NumberOfChannels = -1
    NumberOfClasses = -1
    ImageSize = -1
    LatentDimension = -1
    EpochCount = -1
    RefreshEachStep = -1
    TensorDatasets = None
    SaveCheckpoints = False
    UseSavedModel = False
    CheckpointPath = ""
    LatestCheckpointPath = ""
    LogPath = ""
    OutputDir = ""
    FormatImages = True

    TrainingDataDir = ""
    DatasetSplit = 0

    LRScheduler = ''

    KerasModel = None
    Trainer : bk.BaseKerasModelTrainer = None

    def __init__(self, batchSize, numberOfChannels, numberOfClasses, imageSize, latentDimension, epochCount, refreshEachStep, trainingDataDir, outputDir, saveCheckpoints, useSavedModel, checkpointPath, latestCheckpointPath, logPath, datasetSplit, LRScheduler, formatImages):
        self.BatchSize = batchSize
        self.NumberOfChannels = numberOfChannels
        self.NumberOfClasses = numberOfClasses
        self.ImageSize = imageSize
        self.LatentDimension = latentDimension
        self.EpochCount = epochCount
        self.RefreshEachStep = refreshEachStep
        self.TrainingDataDir = trainingDataDir
        self.OutputDir = outputDir
        self.SaveCheckpoints = saveCheckpoints
        self.UseSavedModel = useSavedModel
        self.CheckpointPath = checkpointPath
        self.LatestCheckpointPath = latestCheckpointPath
        self.LogPath = logPath
        self.DatasetSplit = datasetSplit
        self.LRScheduler = LRScheduler
        self.FormatImages = formatImages

    def SetupModel(self):
        raise Exception("Model setup not implemented")

    def LoadDataset(self):
        dataLoader = dl.DatasetLoader(
            self.TrainingDataDir,
            (self.ImageSize,self.ImageSize),
            self.FormatImages)
        dataLoader.LoadDatasets()
        dataArray = dataLoader.DataSets

        bulkDatasetFormatter = df.BulkDatasetFormatter(dataArray, self.NumberOfClasses,self.BatchSize, self.DatasetSplit)
        self.TensorDatasets = bulkDatasetFormatter.ProcessData()

    def TrainModel(self):
        if self.Trainer == None:
            self.SetupModel()
        if self.UseSavedModel == True:
            checkpointPath = self.__GetCheckpointPath()
            if not checkpointPath:
                print("Checkpoint not found! Training instead")
                self.UseSavedModel = False
        else:
            if self.TensorDatasets == None:
                self.LoadDataset()

        if self.UseSavedModel:
            print("Attempting to load model from checkpoint...")
            self.LoadCheckpoint(checkpointPath)
            print("Checkpoint loaded!")
        else:
            if self.TensorDatasets == None:
                self.LoadDataset()
            self.Trainer.Datasets = self.TensorDatasets
            self.Trainer.TrainModel()

    def __GetCheckpointPath(self):
        if not os.path.exists(self.LatestCheckpointPath):
            return None

        with open(self.LatestCheckpointPath, 'r') as f:
            ckptPath = f.readline().strip()

        if not os.path.exists(f"{ckptPath}.index"):
            return None
        else:
            return ckptPath

    def ProduceOutput(self):
        raise Exception("ML Model not set to produce output!")

    def LoadCheckpoint(self, checkpointPath):
        raise Exception("Checkpoint loading not implemented")
           