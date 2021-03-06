from ProjectTools import AutoPackageInstaller as ap

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
    DatasetSplit = 0

    def __init__(self, images, labels, numberOfLabels, batchSize, datasetSplit):
        self.Images = images
        self.Labels = labels
        self.NumberOfLabels = numberOfLabels
        self.BatchSize = batchSize
        self.DatasetSplit = datasetSplit

    def ProcessData(self):
        # Scale the pixel values to [0, 1] range, add a channel dimension to
        # the images, and one-hot encode the labels.
        self.Images = self.Images.astype("float32") / 255.0
        self.Images = np.reshape(self.Images, (-1, 28, 28, 1))
        self.Labels = tf.keras.utils.to_categorical(self.Labels, self.NumberOfLabels)

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
    DatasetSplit = 0

    def __init__(self, dataArrays, numberOfLabels, batchSize, datasetSplit):
        self.DataArrays = dataArrays
        self.NumberOfLabels = numberOfLabels
        self.BatchSize = batchSize
        self.DatasetSplit = datasetSplit

    def ProcessData(self):
        tensorDatasets = []
        print(f"Converting all data arrays into tensorflow datasets...")
        for dataset in tqdm(iterable=self.DataArrays, total=len(self.DataArrays)):
            (images, labels) = dataset
            datasetFormatter = DatasetFormatter(images, labels, self.NumberOfLabels, self.BatchSize, self.DatasetSplit)
            tensorDatasets.append(self.SplitDataset(datasetFormatter.ProcessData(), len(labels)))
        print(f"A total of {len(tensorDatasets)} have been formatted to tensorflow datasets!")
        return tensorDatasets

    def SplitDataset(self, dataset, dataSize):
        train_size = int(self.DatasetSplit * (dataSize / self.BatchSize))
        test_size = int((1 - self.DatasetSplit) * (dataSize / self.BatchSize))

        train_dataset = dataset.take(train_size)
        test_dataset = dataset.skip(train_size).take(test_size)

        return (train_dataset, test_dataset)