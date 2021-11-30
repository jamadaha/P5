from ProjectTools import AutoPackageInstaller as ap

ap.CheckAndInstall("tqdm")
ap.CheckAndInstall("os")
ap.CheckAndInstall("cv2","opencv-python") #cv2
ap.CheckAndInstall("numpy")

import os
from tqdm import tqdm
import cv2
import numpy as np
from DatasetLoader.FitData import FitData
import tensorflow
from multipledispatch import dispatch
import pathlib

class DiskReader():
    Dir = ""
    LableID = -1
    Images = []
    Labels = []
    DataSize = 0
    ImageSize = ()

    def __init__(self, dir, labelID, imageSize):
        self.Dir = dir
        self.LableID = labelID
        self.ImageSize = imageSize

    def ReadImagesAndLabelsFromDisc(self):
        features, labels = [], []

        if os.path.isdir(self.Dir):
            for img_name in os.listdir(self.Dir):
                img = cv2.imread(os.path.join(self.Dir, img_name), cv2.IMREAD_GRAYSCALE)
                img = cv2.bitwise_not(img)
                img = cv2.resize(img, self.ImageSize)
                features.append(img)
                labels.append(self.LableID)
                self.DataSize += 1
        else:
            return None

        self.Images = np.array(features, dtype=np.float32)
        self.Labels = np.array(labels, dtype=np.float32)

class DatasetLoader():
    TotalImageCount = 0
    TrainDir = ""
    TestDir = ""
    DataSets = []
    ImageSize = ()

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
            dr = DiskReader(dir + dirID, dirID, self.ImageSize)
            dr.ReadImagesAndLabelsFromDisc()
            imageCount += dr.DataSize
            self.DataSets.append((dr.Images,dr.Labels))
        print(f"A total of {imageCount} have been loaded from '{dir}'!")
        self.TotalImageCount += imageCount

    #CLASSIFIER METHODS
    def __CacheDataset(self, dataset: tensorflow.data.Dataset):
        #Cached images are kept in memory after they're loaded - ensures the dataset does not become a bottleneck when training. 
        '''AUTOTUNE = tensorflow.data.experimental.AUTOTUNE
        cached_ds = dataset.cache().prefetch(buffer_size=AUTOTUNE)
        return cached_ds'''
        return dataset

    '''@dispatch(FitData)
    def __NormalizePixelRange(self, dataset: FitData.FitData):
        trainds = self.__NormalizePixelRange(dataset.GetTrainData())
        valds = self.__NormalizePixelRange(dataset.GetValidationData())
        dataset.SetTrainData(trainds)
        dataset.GetValidationData(valds)
        return dataset'''

    @dispatch(tensorflow.data.Dataset)
    def __NormalizePixelRange(self, dataset: tensorflow.data.Dataset):
        normalization_layer = tensorflow.keras.layers.Rescaling(.1/255)
        normalized_ds = dataset.map(lambda x, y: (normalization_layer(x), y))
        image_batch, labels_batch = next(iter(normalized_ds))
        return normalized_ds

    #Method completely loads data from path, creates af FittingData object and preprocesses data for training
    def LoadFittingData(self, path:str, batch_size, img_height, img_width, seed, split) -> FitData: 
        ClassCount = 0
        for entry in os.scandir(self.TrainDir):
            if entry.is_dir():
                ClassCount += 1
        
        td = self.__LoadFromDir(path, batch_size, img_height, img_width, "training", seed, split)
        vd = self.__LoadFromDir(path, batch_size, img_height, img_width, "validation", seed, split)
        td = self.__PreprocessImages(td)
        vd = self.__PreprocessImages(vd)
        fd = FitData(td, vd, ClassCount)
        return fd

    def LoadDataSet(self, path: str, validation_split = 0.2, subset = "validation", seed = 123):
        data_dir = pathlib.Path(path)
        ds = tensorflow.keras.utils.image_dataset_from_directory(
            data_dir, 
            labels="inferred",
            label_mode="int",
            class_names=None,
            validation_split = validation_split,
            subset=subset,
            seed=seed,
            color_mode="rgb",
            batch_size=self.batch_size, 
            image_size=(self.img_height, self.img_width), 
            shuffle=True, 
            interpolation="bilinear", 
            follow_links=False, 
            crop_to_aspect_ratio=False )

        return self.__PreprocessImages(ds)

    #Caches datasets in FittingData for training and validation set and normalizes pixel range
    @dispatch(tensorflow.data.Dataset)
    def __PreprocessImages(self, dataset: tensorflow.data.Dataset):
        norm_data = self.__NormalizePixelRange(dataset)
        norm_data = self.__CacheDataset(norm_data)
        return norm_data

    def __LoadFromDir(self, path: str, batch_size, img_height, img_width, subset, seed, split):
        data_dir = pathlib.Path(path)

        ds = tensorflow.keras.utils.image_dataset_from_directory(data_dir, 
                                                                            labels="inferred", 
                                                                            label_mode="int", 
                                                                            class_names=None, 
                                                                            color_mode="rgb", 
                                                                            batch_size=batch_size, 
                                                                            image_size=(img_height, img_width), 
                                                                            shuffle=True, 
                                                                            seed=seed, 
                                                                            validation_split=split, 
                                                                            subset=subset, 
                                                                            interpolation="bilinear", 
                                                                            follow_links=False, 
                                                                            crop_to_aspect_ratio=False)

        return ds
