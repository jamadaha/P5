

from ProjectTools import AutoPackageInstaller as ap
from ExperimentQueue.Experiment4.ImageNoiseGen import ImageNoiseGen as imgNoiseGen
from Classifier import ClassifierMLModel as cf
from importlib import reload

ap.CheckAndInstall("csv")
ap.CheckAndInstall("tqdm")

import csv
import numpy as np
import random
from tqdm import tqdm

from DatasetLoader import DatasetLoader as DatasetLoader

def getDistribution(csvPath):
    with open(csvPath, newline='') as csvfile:
        csvReader = csv.reader(csvfile, delimiter=',', quotechar='|')
        
        distribution = {}
        for row in csvReader:
            letter = row[0]
            classId = row[1]
            count = row[2]

            distribution[classId] = count
            
        return distribution


class NewDatasetLoader(DatasetLoader.DatasetLoader):
    def __init__(self, trainDir, testDir, imageSize, formatImages):
        super().__init__(trainDir, testDir, imageSize, formatImages)
    
    def __limitDataSetsToDistribution(self, distribution):
        highestInstanceCount = max(distribution)

        for i in tqdm(range(len(self.DataSets)), "Limiting datasets to distribution"):
            label = str(int(self.DataSets[i][1][0])) #first label
            wantedInstanceCount = min(int(distribution[label]), len(self.DataSets[i][0]))
            wantedInstanceCount = max(1, wantedInstanceCount) # Keep at least 1 instace of every class

            self.DataSets[i] = (
                self.DataSets[i][0][:wantedInstanceCount], #images
                self.DataSets[i][1][:wantedInstanceCount]  #labels
                )


    def LoadTrainDatasets(self):
        super().LoadTrainDatasets()

        distribution = getDistribution("../../Data/Distribution.csv")
        self.__limitDataSetsToDistribution(distribution)

        highestInstanceCount = max([len(dataset[0]) for dataset in self.DataSets])
        self.AugmentData(highestInstanceCount)

    def AugmentData(self, wantedMinimumCount):
        random.seed(1234)
        noiseGen = imgNoiseGen()

        highestInstanceCount = max([len(dataset[0]) for dataset in self.DataSets])

        for i in tqdm(range(len(self.DataSets)), "Augmenting datasets"):
            dataset = self.DataSets[i]
            images = dataset[0]
            labels = dataset[1]

            originalImageCount = len(images)

            if originalImageCount == 0:
                print(f"No instances found in dataset: {dataset}")
                continue

            imagesToGenerate = max(0, wantedMinimumCount - originalImageCount)

            newImages = []
            newLabels = []
            for j in range(imagesToGenerate):
                baseImage = dataset[0][j % originalImageCount]
                newImages.append(noiseGen.ApplyNoise(baseImage, random.random()))
                newLabels.append(labels[0])


            self.DataSets[i] = (
                np.append(dataset[0], newImages),
                np.append(dataset[1], newLabels)
                )


class newClassifier(cf.ClassifierMLModel):
    def __init__(self, batchSize, numberOfChannels, numberOfClasses, imageSize, epochCount, refreshEachStep, trainingDataDir, testingDataDir, classifyDir, outputDir, saveCheckpoints, useSavedModel, checkpointPath, latestCheckpointPath, logPath, datasetSplit, LRScheduler, learningRateClass, formatImages, formatClassificationImages):
        import DatasetLoader as dl
        reload(dl.DatasetLoader)
        reload(dl.DatasetLoader.LoadTrainDatasets)
        reload(dl)
        super().__init__(batchSize, numberOfChannels, numberOfClasses, imageSize, epochCount, refreshEachStep, trainingDataDir, testingDataDir, classifyDir, outputDir, saveCheckpoints, useSavedModel, checkpointPath, latestCheckpointPath, logPath, datasetSplit, LRScheduler, learningRateClass, formatImages, formatClassificationImages)

DatasetLoader.DatasetLoader = NewDatasetLoader;

