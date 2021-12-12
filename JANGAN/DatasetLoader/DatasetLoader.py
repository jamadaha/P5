
from ProjectTools import AutoPackageInstaller as ap

ap.CheckAndInstall("tqdm")
ap.CheckAndInstall("os")

from DatasetLoader import DiskReader as ds

import os
from tqdm import tqdm

class DatasetLoader():
    TotalImageCount = 0
    Dir = ""
    DataSets = []
    ImageSize = ()
    NumberOfLabels = 0
    FormatImages = True

    def __init__(self, dir, imageSize, formatImages):
        self.Dir = dir
        self.ImageSize = imageSize
        self.FormatImages = formatImages

    def LoadDatasets(self):
        self.__LoadDataset(self.Dir)

    def __LoadDataset(self,dir):
        dataDir = os.listdir(dir)
        imageCount = 0
        print(f"Loading data from '{dir}'...")
        for dirID in tqdm(iterable=dataDir, total=len(dataDir)):
            dr = ds.DiskReader(dir + dirID, dirID, self.ImageSize, self.FormatImages)
            dr.ReadImagesAndLabelsFromDisc()
            imageCount += dr.DataSize
            self.DataSets.append((dr.Images,dr.Labels))
            self.NumberOfLabels += 1
        print(f"A total of {imageCount} have been loaded from '{dir}'!")
        self.TotalImageCount += imageCount