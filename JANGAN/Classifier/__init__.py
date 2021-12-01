from ProjectTools import AutoPackageInstaller as ap
from ProjectTools import BaseMLModel as bm

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

class Classifier(bm.BaseMLModel):
    ClassifyDir = ""

    AccuracyThreshold = 0

    LearningRateDis = 0

    Classifier = None

    def __init__(self, batchSize, numberOfChannels, numberOfClasses, imageSize, epochCount, refreshEachStep, trainingDataDir, testingDataDir, classifyDir, outputDir, saveCheckpoints, useSavedModel, checkpointPath, latestCheckpointPath, logPath, datasetSplit, LRScheduler, learningRateClass, accuracyThresshold):
        super().__init__(batchSize, numberOfChannels, numberOfClasses, imageSize, None, epochCount, refreshEachStep, trainingDataDir, testingDataDir, outputDir, saveCheckpoints, useSavedModel, checkpointPath, latestCheckpointPath, logPath, datasetSplit, LRScheduler)
        self.ClassifyDir = classifyDir
        self.AccuracyThreshold = accuracyThresshold
        self.LearningRateDis = learningRateClass


    def SetupModel(self):
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

        self.Trainer = ct.ClassifierTrainer(self.Classifier, self.TensorDatasets, self.EpochCount, self.RefreshEachStep, self.SaveCheckpoints, self.CheckpointPath, self.LatestCheckpointPath, self.LogPath)

    def ProduceOutput(self):
        self.UseSavedModel = True
        if self.Classifier == None:
            self.TrainGAN()

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

            print(f"Classifier predicted: {correctPredictions} correct, {incorrectPredictions} incorrect, {((correctPredictions/predictionsCount)*100):.2f}%")

            index += 1

        print(f"Total accuracy of classified dataset: {totalCorrectPredictions} correct, {totalIncorrectPredictions} incorrect, {((totalCorrectPredictions/totalPredictionsCount)*100):.2f}%")