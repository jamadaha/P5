from ProjectTools import AutoPackageInstaller as ap

ap.CheckAndInstall("tensorflow")
ap.CheckAndInstall("numpy")
ap.CheckAndInstall("pathlib")
ap.CheckAndInstall("PIL")
ap.CheckAndInstall("multipledispatch")
ap.CheckAndInstall("matplotlib")

import tensorflow
from tensorflow.keras import utils
import numpy
import os
import PIL
import PIL.Image
import pathlib
from multipledispatch import dispatch
import matplotlib.pyplot as plt
from Classifier.FitData import FitData

class DataLoader(object):
    """A class to handle loading of data for the model. Is responsible for all direct manipulation of the data"""    
    def __init__(self, batch_size: int, img_height: int, img_width: int, seed: int):
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width
        self.seed = seed

    def __CacheDataset(self, dataset: tensorflow.data.Dataset):
        #Cached images are kept in memory after they're loaded - ensures the dataset does not become a bottleneck when training. 
        AUTOTUNE = tensorflow.data.experimental.AUTOTUNE
        cached_ds = dataset.cache().prefetch(buffer_size=AUTOTUNE)
        return cached_ds

    @dispatch(FitData)
    def __NormalizePixelRange(self, dataset: FitData):
        trainds = self.__NormalizePixelRange(dataset.GetTrainData())
        valds = self.__NormalizePixelRange(dataset.GetValidationData())
        dataset.SetTrainData(trainds)
        dataset.GetValidationData(valds)
        return dataset

    @dispatch(tensorflow.data.Dataset)
    def __NormalizePixelRange(self, dataset: tensorflow.data.Dataset):
        normalization_layer = tensorflow.keras.layers.Rescaling(.1/255)
        normalized_ds = dataset.map(lambda x, y: (normalization_layer(x), y))
        image_batch, labels_batch = next(iter(normalized_ds))
        return normalized_ds

    #Method completely loads data from path, creates af FittingData object and preprocesses data for training
    def LoadFittingData(self, path:str): 
        fd = self.__LoadFromDir(path)
        td = self.__PreprocessImages(fd.GetTrainData())
        vd = self.__PreprocessImages(fd.GetValidationData())
        fd.SetTrainData(td)
        fd.SetValidationData(vd)
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

    def __LoadFromDir(self, path: str):
        data_dir = pathlib.Path(path)

        training_ds = tensorflow.keras.utils.image_dataset_from_directory(data_dir, 
                                                                            labels="inferred", 
                                                                            label_mode="int", 
                                                                            class_names=None, 
                                                                            color_mode="rgb", 
                                                                            batch_size=self.batch_size, 
                                                                            image_size=(self.img_height, self.img_width), 
                                                                            shuffle=True, 
                                                                            seed=self.seed, 
                                                                            validation_split=0.2, 
                                                                            subset="training", 
                                                                            interpolation="bilinear", 
                                                                            follow_links=False, 
                                                                            crop_to_aspect_ratio=False)

        validation_ds = tensorflow.keras.utils.image_dataset_from_directory(data_dir, 
                                                                            labels="inferred", 
                                                                            label_mode="int", 
                                                                            class_names=None, 
                                                                            color_mode="rgb", 
                                                                            batch_size=self.batch_size, 
                                                                            image_size=(self.img_height, self.img_width), 
                                                                            shuffle=True, 
                                                                            seed=self.seed, 
                                                                            validation_split=0.2, 
                                                                            subset="validation", 
                                                                            interpolation="bilinear", 
                                                                            follow_links=False, crop_to_aspect_ratio=False)

        
        fitting_data = FitData(training_ds, validation_ds)

        return fitting_data