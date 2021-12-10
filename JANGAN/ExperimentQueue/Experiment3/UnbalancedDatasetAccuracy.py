# Change functions and methods, to fit the goal of the experiment
from CGAN import CGANTrainer as cgt
from Classifier import ClassifierMLModel as cf
from DatasetLoader import DatasetFormatter as dtf

from ProjectTools import AutoPackageInstaller as ap

ap.CheckAndInstall("tensorflow")
ap.CheckAndInstall("time")
ap.CheckAndInstall("csv")
ap.CheckAndInstall("tqdm")

import tensorflow as tf
from tensorflow import keras
import time
import os
import shutil
import tensorboard
import numpy as np
from tqdm import tqdm
from importlib import reload

import csv

batchSize = 0

class newCGANTrainer(cgt.CGANTrainer):
    CSVPath = ""
    CSVData = {}
    CSVDataLabels = {}

    def __init__(self, model, datasets, epochs, refreshUIEachXStep, saveCheckPoints, checkpointPath, latestCheckpointPath, logPath, numberOfClasses, latentDimension, epochImgDir, trackModeCollapse, modeCollapseThreshold):
        super().__init__(model, datasets, epochs, refreshUIEachXStep, saveCheckPoints, checkpointPath, latestCheckpointPath, logPath, numberOfClasses, latentDimension, epochImgDir, trackModeCollapse, modeCollapseThreshold)
        self.CSVPath = "../../Data/Distribution.csv"
        with open(self.CSVPath, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            lineCount = 0
            for row in spamreader:
                if lineCount > 0:
                    self.CSVData[row[1]] = row[2]
                    self.CSVDataLabels[row[1]] = row[0]
                lineCount += 1

    def CreateDataSet(self, dataArray):
        index = 1
        (returnTrainSet, returnTestSet) = self.TakePartOfDataset(0, dataArray[0])
        for data in dataArray[1:]:
            (addTrainSet, addTestSet) = self.TakePartOfDataset(index, data)
            returnTrainSet = returnTrainSet.concatenate(addTrainSet)
            returnTestSet = returnTestSet.concatenate(addTestSet)
            index += 1
        
        returnTrainSet = returnTrainSet.shuffle(buffer_size=1024)
        returnTestSet = returnTestSet.shuffle(buffer_size=1024)
        return (returnTrainSet, returnTestSet)

    def TakePartOfDataset(self, index, data):
        takeSize = int(self.CSVData[str(index)])
        if takeSize == 0:
            takeSize = 1
        (returnTrainSet, returnTestSet) = data
        returnTrainSet = returnTrainSet.shuffle(buffer_size=1024).take(takeSize)
        returnTestSet = returnTestSet.shuffle(buffer_size=1024).take(takeSize)

        global batchSize
        data = (returnTrainSet.batch(batchSize), returnTestSet.batch(batchSize))

        return data

class newClassifier(cf.ClassifierMLModel):
    def __init__(self, batchSize, numberOfChannels, numberOfClasses, imageSize, epochCount, refreshEachStep, trainingDataDir, classifyDir, outputDir, saveCheckpoints, useSavedModel, checkpointPath, latestCheckpointPath, logPath, datasetSplit, LRScheduler, learningRateClass, formatImages, formatClassificationImages):
        super().__init__(batchSize, numberOfChannels, numberOfClasses, imageSize, epochCount, refreshEachStep, trainingDataDir, classifyDir, outputDir, saveCheckpoints, useSavedModel, checkpointPath, latestCheckpointPath, logPath, datasetSplit, LRScheduler, learningRateClass, formatImages, formatClassificationImages)
        import DatasetLoader as dl
        reload(dl.DatasetFormatter)
        reload(dl.DatasetLoader)
        reload(dl.DiskReader)
        reload(dl)

class newDatasetFormatter(dtf.DatasetFormatter):
    def ProcessData(self):
        # Scale the pixel values to [0, 1] range, add a channel dimension to
        # the images, and one-hot encode the labels.
        self.Images = self.Images.astype("float32") / 255.0
        self.Images = np.reshape(self.Images, (-1, 28, 28, 1))
        self.Labels = tf.keras.utils.to_categorical(self.Labels, self.NumberOfLabels)

        # Create tf.data.Dataset.
        dataset = tf.data.Dataset.from_tensor_slices((self.Images, self.Labels))

        # Shuffle and batch
        dataset = dataset.shuffle(buffer_size=1024)

        global batchSize
        batchSize = self.BatchSize

        return dataset

cgt.CGANTrainer = newCGANTrainer
cf.ClassifierMLModel = newClassifier
dtf.DatasetFormatter = newDatasetFormatter