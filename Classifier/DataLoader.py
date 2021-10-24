import os
import tensorflow
class DataLoader(object):
    """description of class"""
    def __init__(self, path: str):
        self.path = path

    def load_from_dir(self, path: str):
        self.dataset = tensorflow.keras.utils.image_dataset_from_directory(path, labels='inferred')
        return dataset

    def load_from_dir(self):
        self.dataset = tensorflow.keras.utils.image_dataset_from_directory(self.path, labels='inferred')
        return dataset

    def set_path(path: str):
        self.path = path


    








