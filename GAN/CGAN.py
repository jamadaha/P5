import sys
sys.path.append('./ProjectTools')

import DatasetLoader as dl
import DatasetFormatter as df
import CGANKerasModel as km
import LayerDefinition as ld

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
batch_size = 32
num_channels = 1
num_classes = 10
image_size = 28
latent_dim = 128
epoch_count = 25
refreshEachStep = 20

generator_in_channels = latent_dim + num_classes
discriminator_in_channels = num_channels + num_classes
print(generator_in_channels, discriminator_in_channels)

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
    discriminator=layerDefiniton.GetDiscriminator(), generator=layerDefiniton.GetGenerator(), latent_dim=latent_dim, imageSize=image_size, numberOfClasses=num_classes
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





























# interpolate
sentinel = True
while(sentinel):
    Question = input("Enter a new index to generate (0-" + str(num_classes)+ ")(type N to exit):")
    if Question == "N":
        sentinel = False
        break

    value = int(Question)

    # Choose the number of intermediate images that would be generated in
    # between the interpolation + 2 (start and last images).
    num_interpolation = 10  # @param {type:"integer"}

    # Sample noise for the interpolation.
    interpolation_noise = tf.random.normal(shape=(1, latent_dim))
    interpolation_noise = tf.repeat(interpolation_noise, repeats=num_interpolation)
    interpolation_noise = tf.reshape(interpolation_noise, (num_interpolation, latent_dim))


    def interpolate_class(first_number, second_number):
        # Convert the start and end labels to one-hot encoded vectors.
        first_label = keras.utils.to_categorical([first_number], num_classes)
        second_label = keras.utils.to_categorical([second_number], num_classes)
        first_label = tf.cast(first_label, tf.float32)
        second_label = tf.cast(second_label, tf.float32)

        # Calculate the interpolation vector between the two labels.
        percent_second_label = tf.linspace(0, 1, num_interpolation)[:, None]
        percent_second_label = tf.cast(percent_second_label, tf.float32)
        interpolation_labels = (
            first_label * (1 - percent_second_label) + second_label * percent_second_label
        )

        # Combine the noise and the labels and run inference with the generator.
        noise_and_labels = tf.concat([interpolation_noise, interpolation_labels], 1)
        fake = trained_gen.predict(noise_and_labels)
        return fake

    fake_images = interpolate_class(value, value)
    fake_images *= 255.0

    #Generate Gif
    imageio.mimsave('out.gif', fake_images)