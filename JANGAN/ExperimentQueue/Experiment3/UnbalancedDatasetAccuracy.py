# Change functions and methods, to fit the goal of the experiment
from CGAN import CGANTrainer as cgt
from DatasetLoader import DatasetFormatter as dtf

from ProjectTools import AutoPackageInstaller as ap

ap.CheckAndInstall("tensorflow")
ap.CheckAndInstall("time")
ap.CheckAndInstall("csv")
ap.CheckAndInstall("tqdm")

import tensorflow as tf
import numpy as np
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
        global batchSize
        takeSize = int(self.CSVData[str(index)])
        if takeSize < batchSize:
            takeSize = batchSize
        takeSize = int(takeSize / batchSize)
        (returnTrainSet, returnTestSet) = data

        returnTrainSet = returnTrainSet.shuffle(buffer_size=1024).take(takeSize)
        returnTestSet = returnTestSet.shuffle(buffer_size=1024).take(takeSize)

        return (returnTrainSet, returnTestSet)

class newDatasetFormatter(dtf.DatasetFormatter):
    def ProcessData(self):
        global batchSize
        batchSize = self.BatchSize
        return super().ProcessData()
   
cgt.CGANTrainer = newCGANTrainer
dtf.DatasetFormatter = newDatasetFormatter