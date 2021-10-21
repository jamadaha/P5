import AutoPackageInstaller as ap

ap.CheckAndInstall("numpy")
ap.CheckAndInstall("tensorflow")
ap.CheckAndInstall("tqdm")

import tensorflow as tf
import numpy as np
from tqdm import tqdm

# For formatting a single dataset of images and labels
class DatasetFormatter():
    Images = []
    Labels = []
    NumberOfLabels = -1
    BatchSize = -1
    ImageSize = ()

    def __init__(self, images, labels, numberOfLabels, batchSize, imageSize):
        self.Images = images
        self.Labels = labels
        self.NumberOfLabels = numberOfLabels
        self.BatchSize = batchSize
        self.ImageSize = imageSize

    def ProcessData(self):
        # Scale the pixel values to [0, 1] range, add a channel dimension to
        # the images, and one-hot encode the labels.
        self.Images = tf.image.resize(self.Images, np.asarray(self.ImageSize))
        self.Images = self.Images.astype("float32") / 255.0
        self.Images = np.reshape(self.Images, (-1, 28, 28, 1))
        self.Labels = keras.utils.to_categorical(self.Labels, self.NumberOfLabels)

        # Create tf.data.Dataset.
        dataset = tf.data.Dataset.from_tensor_slices((self.Images, self.Labels))

        # Shuffle and batch
        dataset = dataset.shuffle(buffer_size=1024).batch(self.BatchSize)

        return dataset

# For formatting an array of datasets 
class BulkDatasetFormatter():
    DataArrays = []
    NumberOfLabels = -1
    BatchSize = -1
    ImageSize = ()

    def __init__(self, dataArrays, numberOfLabels, batchSize, imageSize):
        self.DataArrays = dataArrays
        self.NumberOfLabels = numberOfLabels
        self.BatchSize = batchSize
        self.ImageSize = imageSize

    def ProcessData(self):
        tensorDatasets = []
        print(f"Converting all data arrays into tensorflow datasets...")
        for dataset in tqdm(iterable=self.DataArrays, total=len(self.DataArrays)):
            (images, labels) = dataset
            datasetFormatter = DatasetFormatter(images, labels, self.NumberOfLabels, self.BatchSize, self.ImageSize)
            tensorDatasets.append(df.ProcessData())
        print(f"A total of {len(tensorDatasets)} have been formatted to tensorflow datasets!")
        return tensorDatasets