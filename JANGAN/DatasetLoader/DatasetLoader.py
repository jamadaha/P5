
from ProjectTools import AutoPackageInstaller as ap

ap.CheckAndInstall("tqdm")
ap.CheckAndInstall("os")

from DatasetLoader import DiskReader as ds

import os
from tqdm import tqdm

class DatasetLoader():
    TotalImageCount = 0
    TrainDir = ""
    TestDir = ""
    DataSets = []
    ImageSize = ()
    NumberOfLabels = 0

    def __init__(self, trainDir, testDir, imageSize):
        self.TrainDir = trainDir
        self.TestDir = testDir
        self.ImageSize = imageSize

    def LoadTestDatasets(self):
        self.LoadDataset(self.TestDir)

    def LoadTrainDatasets(self):
        self.LoadDataset(self.TrainDir)

    def LoadDataset(self,dir):
        dataDir = os.listdir(dir)
        imageCount = 0
        print(f"Loading data from '{dir}'...")
        for dirID in tqdm(iterable=dataDir, total=len(dataDir)):
            dr = ds.DiskReader(dir + dirID, dirID, self.ImageSize)
            dr.ReadImagesAndLabelsFromDisc()
            imageCount += dr.DataSize
            self.DataSets.append((dr.Images,dr.Labels))
            self.NumberOfLabels += 1
        print(f"A total of {imageCount} have been loaded from '{dir}'!")
        self.TotalImageCount += imageCount