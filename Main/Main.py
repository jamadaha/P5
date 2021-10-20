from numpy.core.records import array
import tensorflow as tf
print("Version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time

from IPython import display

import pandas as pd

from ProjectTools import ConfigHelper as cfg
csvPath = cfg.GetStringValue("DATAGENERATOR","CSVFileName")
imageOutputPath = cfg.GetStringValue("GAN","ImageOutputPath")
os.makedirs(imageOutputPath, exist_ok=True)
checkpointPath = cfg.GetStringValue("GAN","CheckpointPath")
os.makedirs(checkpointPath, exist_ok=True)

from GAN import GAN as GAN
gan = GAN.GAN(checkpointPath)


csvData = pd.read_csv(csvPath,
    skiprows=1,
    #nrows=1000,
    names=["Letter","Path"])



IMAGE_RESOLUTION = (28, 28)

def ImageFromRow(row):
    fileContent = tf.io.read_file(row["Path"])
    jpg = tf.io.decode_jpeg(fileContent)
    jpg = tf.image.rgb_to_grayscale(jpg)
    jpg = tf.image.resize(jpg, IMAGE_RESOLUTION)
    return jpg
def LabelsFromRow(row):
    out = row.copy()
    del out["image_id"]
    return out

def normalizeImage(jpg):
    jpgMatrix = np.reshape(jpg, IMAGE_RESOLUTION[0], IMAGE_RESOLUTION[1], 1).astype('float32')
    jpgMatrix = (jpgMatrix - 127.5) / 127.5  # Normalize the images to [-1, 1]

def process_csv_data(row):
    return {
        "image": normalizeImage(ImageFromRow(row)),
        "labels": LabelsFromRow(row)
    }

trainData = np.array([ImageFromRow(row[1]) for row in csvData.iterrows()])


BUFFER_SIZE = 60000
BATCH_SIZE = 256

# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(trainData).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


def saveImageMatrix(images, path): # Only 4x4 atm
    #firstImages = images[0, :, :, 1]
    for i in range(len(images)):
        plt.subplot(4, 4, i+1)
        img = images[0][i, :, :, 0]
        #img = np.reshape(images[i], 28, 28, 1)
        plt.imshow(img * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig(path)


noise_dim = 100
num_examples_to_generate = 4*4
randomNums = tf.random.normal([num_examples_to_generate, num_examples_to_generate, noise_dim])


def generateAndSaveImage(epoch):
    generatedImages = [gan.Generate(i) for i in randomNums]
    #generatedImages = [[gan.Generate() for i in randomNums] for j in range(0,16)]
    path = os.path.join(imageOutputPath, 'image_at_epoch_{:04d}.png'.format(epoch))
    saveImageMatrix(generatedImages, path)


def train(dataset, epochs):
  for epoch in range(epochs):
    start = time.time()

    for image_batch in dataset:
        gan.Train(image_batch, image_batch)
        #train_step(image_batch)


    print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

    # Save the model every 15 epochs
    if (epoch + 1) % 15 == 0:
        gan.SaveCheckpoint('ckpt_at_epoch_{:04d}'.format(epoch))
        generateAndSaveImage(epoch)

#Train Model
train(dataset=train_dataset, epochs=2000)

