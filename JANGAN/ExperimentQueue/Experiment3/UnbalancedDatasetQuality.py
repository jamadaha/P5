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
    
    def __limitDataSetsToDistribution(self, distribution):
        for i in tqdm(range(len(self.DataSets)), "Limiting datasets to distribution"):
            label = str(int(self.DataSets[i][1][0])) #first label
            wantedInstanceCount = min(int(distribution[label]), len(self.DataSets[i][0]))
            wantedInstanceCount = max(1, wantedInstanceCount) # Keep at least 1 instace of every class

            self.DataSets[i] = (
                self.DataSets[i][0][:wantedInstanceCount], #images
                self.DataSets[i][1][:wantedInstanceCount]  #labels
                )


    def LoadDatasets(self):
        super().LoadDatasets()

        distribution = getDistribution("../../Data/Distribution.csv")
        self.__limitDataSetsToDistribution(distribution)


class newClassifier(cf.ClassifierMLModel):
    def __init__(self, *args, **kwargs):
        import DatasetLoader as dl
        reload(dl.DatasetLoader)
        reload(dl)
        super().__init__(*args, **kwargs)

DatasetLoader.DatasetLoader = NewDatasetLoader;
cf.ClassifierMLModel = newClassifier;

