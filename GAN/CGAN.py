import sys
sys.path.append('./ProjectTools')

import ConfigHelper as cfg

import DatasetLoader as dl
import DatasetFormatter as df
import CGANKerasModel as km
import LayerDefinition as ld
import LetterProducer as lp

from tensorflow import keras
from tensorflow.keras import layers

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import imageio
import random
import os
from PIL import Image
from tqdm import tqdm
import shutil
import time

#Constants
batch_size = cfg.GetIntValue("CGAN", "BatchSize")
num_channels = 1
num_classes = cfg.GetIntValue("CGAN", "NumberOfClasses")
image_size = cfg.GetIntValue("CGAN", "ImageSize")
latent_dim = cfg.GetIntValue("CGAN", "LatentDimension")
epoch_count = cfg.GetIntValue("CGAN", "EpochCount")
refreshEachStep = cfg.GetIntValue("CGAN", "RefreshUIEachXIteration")
imageCountToProduce = cfg.GetIntValue("CGAN", "NumberOfFakeImagesToOutput")

generator_in_channels = latent_dim + num_classes
discriminator_in_channels = num_channels + num_classes

def train(allDatasets, gan, epochs):
    print("Training started")
    for epoch in range(epochs):
        start = time.time()

        print(f"Epoch {epoch + 1} of {epochs} is in progress...")
        epochDataset = CreateDataSet(allDatasets)
        itemCount = tf.data.experimental.cardinality(epochDataset).numpy()
        count = 0
        epochTime = time.time()
        for image_batch in epochDataset:
            if count % refreshEachStep == 0:
                returnVal = gan.train_step(image_batch)
                g_loss = float(returnVal['g_loss'])
                d_loss = float(returnVal['d_loss'])
                now = time.time()
                estRemainTime = ((now - epochTime) / refreshEachStep) * (itemCount - count)
                epochTime = now
                print(f"Generator loss: {g_loss:.4f}. Discriminator loss: {d_loss:.4f}. Progress: {((count/itemCount)*100):.2f}%. Est time left: {GetDatetimeFromSeconds(estRemainTime)}    ", end="\r")
            else:
                gan.train_step(image_batch)
            count += 1

        totalEpochTime = time.time()-start
        print("")
        print("Done!")
        print(f"Time for epoch {epoch + 1} is {GetDatetimeFromSeconds(totalEpochTime)}. Est time remaining for training is {GetDatetimeFromSeconds(totalEpochTime*(epochs-(epoch + 1)))}")

def CreateDataSet(dataArray):
    returnSet = dataArray[0]
    for data in dataArray[1:]:
        returnSet = returnSet.concatenate(data)
    return returnSet.shuffle(buffer_size=1024)

def GetDatetimeFromSeconds(seconds):
    return time.strftime("%H:%M:%S", time.gmtime(seconds))

# Setup the CGAN
layerDefiniton = ld.LayerDefinition(discriminator_in_channels,generator_in_channels)

cond_gan = km.ConditionalGAN(
    discriminator=layerDefiniton.GetDiscriminator(), 
    generator=layerDefiniton.GetGenerator(), 
    latent_dim=latent_dim, 
    imageSize=image_size, 
    numberOfClasses=num_classes
)
cond_gan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
    g_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
    loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
)

# Load dataset
dataLoader = dl.DatasetLoader('../Data/Output/','',(image_size,image_size))
dataLoader.LoadTrainDatasets()
dataArray = dataLoader.DataSets

bulkDatasetFormatter = df.BulkDatasetFormatter(dataArray, num_classes,batch_size)
tensorDatasets = bulkDatasetFormatter.ProcessData();

# Train the CGAN
train(tensorDatasets,cond_gan,epoch_count)
trained_gen = cond_gan.generator

# Use the trained generator
sentinel = True
while(sentinel):
    Question = input(f"Enter a new index to generate (0-{num_classes - 1}))(type N to exit):")
    if Question == "N":
        sentinel = False
        break

    value = int(Question)

    if value >= num_classes:
        print(f"Please write numbers within 0-{num_classes - 1}")
        continue
    if value < 0:
        print(f"Please write numbers within 0-{num_classes - 1}")
        continue

    letterProducer = lp.LetterProducer(trained_gen, num_classes, latent_dim)

    images = letterProducer.GenerateLetter(value, 10)
    letterProducer.SaveImagesAsGif(images)