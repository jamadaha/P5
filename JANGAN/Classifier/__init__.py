from ProjectTools import CSVLogger
from ProjectTools import AutoPackageInstaller as ap

ap.CheckAndInstall("tensorflow")
ap.CheckAndInstall("tqdm")
ap.CheckAndInstall("numpy")

import tensorflow as tf
from tensorflow import keras
import os
import numpy as np

from DatasetLoader import DatasetLoader as dl
from DatasetLoader import DatasetFormatter as df
from Classifier import ClassifierKerasModel as cm
from Classifier import LayerDefinition as ld
from Classifier import ClassifierTrainer as ct

class Classifier():
    BatchSize = -1
    NumberOfChannels = -1
    NumberOfClasses = -1
    ImageSize = -1
    EpochCount = -1
    RefreshEachStep = -1
    TensorDatasets = None
    SaveCheckpoints = True
    UseSavedModel = False
    CheckpointPath = ""
    LatestCheckpointPath = ""
    LogPath = ""
    Logger = None
    ClassifyDir = ""

    TrainingDataDir = ""
    TestingDataDir = ""
    DatasetSplit = 0

    AccuracyThreshold = 0

    LRScheduler = ''
    LearningRateClass = 0.0

    Classifier = None
    DataLoader = None

    def __init__(self, batchSize, numberOfChannels, numberOfClasses, imageSize, epochCount, refreshEachStep, trainingDataDir, testingDataDir, classifyDir, saveCheckpoints, useSavedModel, checkpointPath, latestCheckpointPath, logPath, datasetSplit, LRScheduler, learningRateClass, accuracyThresshold):
        self.BatchSize = batchSize
        self.NumberOfChannels = numberOfChannels
        self.NumberOfClasses = numberOfClasses
        self.ImageSize = imageSize
        self.EpochCount = epochCount
        self.RefreshEachStep = refreshEachStep
        self.TrainingDataDir = trainingDataDir
        self.TestingDataDir = testingDataDir
        self.ClassifyDir = classifyDir
        self.SaveCheckpoints = saveCheckpoints
        self.UseSavedModel = useSavedModel
        self.CheckpointPath = checkpointPath
        self.LatestCheckpointPath = latestCheckpointPath
        self.LogPath = logPath
        self.Logger = CSVLogger.CSVLogger(logPath, 'TestData')
        self.Logger.InitCSV(['Index', 'Correct', 'Incorrect', 'CorrectPercentage'])
        self.DatasetSplit = datasetSplit
        self.LRScheduler = LRScheduler
        self.LearningRateClass = learningRateClass
        self.AccuracyThreshold = accuracyThresshold

    def SetupClassifier(self):
        layerDefiniton = ld.LayerDefinition(self.NumberOfClasses)

        self.Classifier = cm.ClassifierModel(
            classifier=layerDefiniton.GetClassifier(), 
            imageSize=self.ImageSize, 
            numberOfClasses=self.NumberOfClasses,
            accuracyThreshold=self.AccuracyThreshold
        )

        if self.LRScheduler == 'Constant':
            self.Classifier.compile(
                optimizer=keras.optimizers.Adam(learning_rate=self.LearningRateClass),
                loss_fn=keras.losses.CategoricalCrossentropy(from_logits=True),
            )  
        elif self.LRScheduler == 'ExponentialDecay':
            classSchedule = keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=self.LearningRateClass,
                decay_steps=10000,
                decay_rate=0.9
            )

            self.Classifier.compile(
                optimizer=keras.optimizers.Adam(learning_rate=classSchedule),
                loss_fn=keras.losses.CategoricalCrossentropy(from_logits=True),
            )  

    def LoadDataset(self):
        if self.UseSavedModel:
            print("Assuming checkpoint exists. Continuing without loading data...")
            return

        dataLoader = dl.DatasetLoader(
            self.TrainingDataDir,
            self.TestingDataDir,
            (self.ImageSize,self.ImageSize))
        dataLoader.LoadTrainDatasets()
        dataArray = dataLoader.DataSets

        bulkDatasetFormatter = df.BulkDatasetFormatter(dataArray, self.NumberOfClasses,self.BatchSize, self.DatasetSplit)
        self.TensorDatasets = bulkDatasetFormatter.ProcessData()

    def TrainClassifier(self):
        checkpointPath = self.__GetCheckpointPath()
        if not checkpointPath:
            print("Checkpoint not found! Training instead")
            self.UseSavedModel = False
            if self.TensorDatasets == None:
                self.LoadDataset()

        classifierTrainer = ct.ClassifierTrainer(self.Classifier, self.TensorDatasets, self.EpochCount, self.RefreshEachStep, self.SaveCheckpoints, self.CheckpointPath, self.LatestCheckpointPath, self.LogPath)

        if self.UseSavedModel:
            print("Attempting to load Classifier model from checkpoint...")
            classifierTrainer.Classifier.load_weights(checkpointPath)
            print("Checkpoint loaded!")
        else:
            classifierTrainer.TrainClassifier()

    def ClassifyData(self):
        dataLoader = dl.DatasetLoader(
            self.ClassifyDir,
            "",
            (self.ImageSize,self.ImageSize))
        dataLoader.DataSets = []
        dataLoader.LoadTrainDatasets()
        dataArray = dataLoader.DataSets

        totalCorrectPredictions = 0
        totalIncorrectPredictions = 0
        totalPredictionsCount = 0
        index = 0
        for data in dataArray:
            print(f"Predictions for index {index}")
            (images, labels) = data
            datasetFormatter = df.DatasetFormatter(images, labels, self.NumberOfClasses, self.BatchSize, 1)
            classifyData = datasetFormatter.ProcessData()

            probability_model = tf.keras.Sequential([self.Classifier.classifier, tf.keras.layers.Softmax()])
            correctPredictions = 0
            incorrectPredictions = 0
            predictionsCount = 0
            for (images, labels) in classifyData:
                predictions = self.Classifier.classifier(images, training=False)
                for prediction in predictions:
                    predictedClass = np.argmax(prediction)
                    if predictedClass == index:
                        correctPredictions += 1
                        totalCorrectPredictions += 1
                    else:
                        incorrectPredictions += 1
                        totalIncorrectPredictions += 1
                    predictionsCount += 1
                    totalPredictionsCount += 1

            correctPercentage = (correctPredictions/predictionsCount)*100
            print(f"Classifier predicted: {correctPredictions} correct, {incorrectPredictions} incorrect, {(correctPercentage):.2f}%")

            self.__LogData(index, correctPredictions, predictionsCount, correctPercentage)

            index += 1

        print(f"Total accuracy of classified dataset: {totalCorrectPredictions} correct, {totalIncorrectPredictions} incorrect, {((totalCorrectPredictions/totalPredictionsCount)*100):.2f}%")


    def __GetCheckpointPath(self):
        pass
        #if not os.path.exists(self.LatestCheckpointPath):
        #    return None

        #with open(self.LatestCheckpointPath, 'r') as f:
        #    ckptPath = f.readline().strip()

        #if not os.path.exists(f"{ckptPath}.index"):
        #    return None
        #else:
        #    return ckptPath

    def __LogData(self, index, correct, incorrect, correctPercentage):
        self.Logger.AppendToCSV([index, correct, incorrect, correctPercentage])
