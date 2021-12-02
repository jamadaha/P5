# Change functions and methods, to fit the goal of the experiment

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
from ProjectTools import Logger as lgr
import tensorboard
import numpy as np
from tqdm import tqdm

import csv

batchSize = 0

from CGAN import CGANTrainer as cg
from DatasetLoader import DatasetFormatter as dtf

class newCGANTrainer(cg.CGANTrainer):
    CSVPath = ""
    CSVData = {}
    CSVDataLabels = {}
    HighestValue = 0

    def __init__(self, batchSize, numberOfChannels, numberOfClasses, imageSize, latentDimension, epochCount, refreshEachStep, imageCountToProduce, trainingDataDir, testingDataDir, outputDir, epochImgDir, saveCheckpoints, useSavedModel, checkpointPath, latestCheckpointPath, logPath, datasetSplit, LRScheduler, learningRateDis, learningRateGen):
        super().__init__(batchSize, numberOfChannels, numberOfClasses, imageSize, latentDimension, epochCount, refreshEachStep, imageCountToProduce, trainingDataDir, testingDataDir, outputDir, epochImgDir, saveCheckpoints, useSavedModel, checkpointPath, latestCheckpointPath, logPath, datasetSplit, LRScheduler, learningRateDis, learningRateGen)
        self.CSVPath = "../../Data/Distribution.csv"
        with open(self.CSVPath, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            lineCount = 0
            for row in spamreader:
                if lineCount > 0:
                    self.CSVData[row[1]] = row[2]
                    self.CSVDataLabels[row[1]] = row[0]
                    if int(row[2]) > HighestValue:
                        HighestValue = int(row[2])
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
        (returnTrainSet, returnTestSet) = data
        originalReturnTrainSetSize = self.GetDatasetSize(returnTrainSet)

        if takeSize > originalReturnTrainSetSize:
            while takeSize > self.GetDatasetSize(returnTrainSet):
                returnTrainSet.append(self.DuplicateImage(returnTrainSet, originalReturnTrainSetSize))
            returnTrainSet = returnTrainSet.shuffle(buffer_size=1024)
        else:
            returnTrainSet = returnTrainSet.shuffle(buffer_size=1024).take(HighestValue - takeSize)

        #returnTestSet = returnTestSet.shuffle(buffer_size=1024).take(HighestValue - takeSize)

        #print(f"Amount of label '{self.CSVDataLabels[str(index)]}' train set: {self.GetDatasetSize(returnTrainSet)}")
        #print(f"Amount of label '{self.CSVDataLabels[str(index)]}' test set: {self.GetDatasetSize(returnTestSet)}")

        global batchSize
        data = (returnTrainSet.batch(batchSize), returnTestSet.batch(batchSize))

        return data

    def DuplicateImage(self, dataset, size):
        return dataset[0]

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
dtf.DatasetFormatter = newDatasetFormatter