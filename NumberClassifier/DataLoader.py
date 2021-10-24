import tensorflow
import numpy
import os
import PIL
import PIL.Image
import pathlib
import matplotlib.pyplot as plt

class DataLoader(object):
    """A class to handle loading of data for the model. Is responsible for all direct manipulation of the data"""    
    def __init__(self):
        #self.path = pathlib.Path(path)
        self.training_ds: tensorflow.data.Dataset
        self.validation_ds: tensorflow.data.Dataset

        self.batch_size = 32
        self.img_height = 180
        self.img_width = 180
        self.seed = 123

    def set_path(path: str):
        self.path = path

    def get_training_ds(self):
        return self.training_ds

    def get_validation_ds(self):
        return self.validation_ds

    def load_from_dir(self, path: str):
        data_dir = pathlib.Path(path)
        self.training_ds = tensorflow.keras.utils.image_dataset_from_directory(data_dir, 
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

        self.validation_ds = tensorflow.keras.utils.image_dataset_from_directory(data_dir, 
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
                                                                           follow_links=False, 
                                                                           crop_to_aspect_ratio=False)


        #Changes the pixel range to [0,1]
    def normalize_pixel_range(self, dataset: tensorflow.data.Dataset):
        normalization_layer = tensorflow.keras.layers.Rescaling(.1/255)
        normalized_ds = dataset.map(lambda x, y: (normalization_layer(x), y))
        image_batch, labels_batch = next(iter(normalized_ds))
        return normalized_ds

    def preprocess_images(self, dataset: tensorflow.data.Dataset):
        self.normalize_pixel_range(dataset)
        
        #Cached images are kept in memory after they're loaded - ensures the dataset does not become a bottleneck when training. 
        AUTOTUNE = tensorflow.data.AUTOTUNE
        cached_ds = dataset.cache().prefetch(buffer_size=AUTOTUNE)
        return cached_ds

        



    








