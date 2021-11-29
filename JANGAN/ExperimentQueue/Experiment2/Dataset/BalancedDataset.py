# Change functions and methods, to fit the goal of the experiment
from CGAN import CGANTrainer as cgt
from CGAN import DatasetFormatter as dtf

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

class newCGANTrainer(cgt.CGANTrainer):
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
        (returnTrainSet, returnTestSet) = data
        returnTrainSet = returnTrainSet.shuffle(buffer_size=1024)
        returnTestSet = returnTestSet.shuffle(buffer_size=1024)

        #print(f"Amount of label '{self.CSVDataLabels[str(index)]}' train set: {self.GetDatasetSize(returnTrainSet)}")
        #print(f"Amount of label '{self.CSVDataLabels[str(index)]}' test set: {self.GetDatasetSize(returnTestSet)}")

        global batchSize
        data = (returnTrainSet.batch(batchSize), returnTestSet.batch(batchSize))

        return data

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
