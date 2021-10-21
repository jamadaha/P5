import sys
sys.path.append('./ProjectTools')

import DatasetLoader as dl
import DatasetFormatter as df

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

# We'll use all the available examples from both the training and test
# sets.

# setup the class
class ConditionalGAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super(ConditionalGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.gen_loss_tracker = keras.metrics.Mean(name="generator_loss")
        self.disc_loss_tracker = keras.metrics.Mean(name="discriminator_loss")

    @property
    def metrics(self):
        return [self.gen_loss_tracker, self.disc_loss_tracker]

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(ConditionalGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, data):
         # Unpack the data.
        real_images, one_hot_labels = data

        # Add dummy dimensions to the labels so that they can be concatenated with
        # the images. This is for the discriminator.
        image_one_hot_labels = one_hot_labels[:, :, None, None]
        image_one_hot_labels = tf.repeat(
            image_one_hot_labels, repeats=[image_size * image_size]
        )
        image_one_hot_labels = tf.reshape(
            image_one_hot_labels, (-1, image_size, image_size, num_classes)
        )

        # Sample random points in the latent space and concatenate the labels.
        # This is for the generator.
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        random_vector_labels = tf.concat(
            [random_latent_vectors, one_hot_labels], axis=1
        )

        # Decode the noise (guided by labels) to fake images.
        generated_images = self.generator(random_vector_labels)

        # Combine them with real images. Note that we are concatenating the labels
        # with these images here.
        fake_image_and_labels = tf.concat([generated_images, image_one_hot_labels], -1)
        real_image_and_labels = tf.concat([real_images, image_one_hot_labels], -1)
        combined_images = tf.concat(
            [fake_image_and_labels, real_image_and_labels], axis=0
        )

        # Assemble labels discriminating real from fake images.
        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )

        # Train the discriminator.
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        # Sample random points in the latent space.
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        random_vector_labels = tf.concat(
            [random_latent_vectors, one_hot_labels], axis=1
        )

        # Assemble labels that say "all real images".
        misleading_labels = tf.zeros((batch_size, 1))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tf.GradientTape() as tape:
            fake_images = self.generator(random_vector_labels)
            fake_image_and_labels = tf.concat([fake_images, image_one_hot_labels], -1)
            predictions = self.discriminator(fake_image_and_labels)
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        # Monitor loss.
        self.gen_loss_tracker.update_state(g_loss)
        self.disc_loss_tracker.update_state(d_loss)
        return {
            "g_loss": self.gen_loss_tracker.result(),
            "d_loss": self.disc_loss_tracker.result(),
        }

# Generator and Discriminator:
# Create the discriminator.
discriminator = keras.Sequential(
    [
        keras.layers.InputLayer((28, 28, discriminator_in_channels)),
        layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.GlobalMaxPooling2D(),
        layers.Dense(1),
    ],
    name="discriminator",
)

# Create the generator.
generator = keras.Sequential(
    [
        keras.layers.InputLayer((generator_in_channels,)),
        # We want to generate 128 + num_classes coefficients to reshape into a
        # 7x7x(128 + num_classes) map.
        layers.Dense(7 * 7 * generator_in_channels),
        layers.LeakyReLU(alpha=0.2),
        layers.Reshape((7, 7, generator_in_channels)),
        layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(1, (7, 7), padding="same", activation="sigmoid"),
    ],
    name="generator",
)

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
    return time.strftime("%H:%M:%S", time.gmtime(n))

# Train the gan:
cond_gan = ConditionalGAN(
    discriminator=discriminator, generator=generator, latent_dim=latent_dim
)
cond_gan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
    g_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
    loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
)

# Load dataset

#(trainX, trainY), (testX, testY) = keras.datasets.mnist.load_data()
#all_digits = np.concatenate([trainX, testX])
#all_labels = np.concatenate([trainY, testY])

#all_digits = all_digits.astype("float32") / 255.0
#all_digits = np.reshape(all_digits, (-1, 28, 28, 1))
#all_labels = keras.utils.to_categorical(all_labels, num_classes)

#dataset = tf.data.Dataset.from_tensor_slices((all_digits, all_labels))
#dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)

#print(f"Shape of training images: {all_digits.shape}")
#print(f"Shape of training labels: {all_labels.shape}")

#allDatasets.append(dataset)

#totalImageCount = 0
#dataDir = os.listdir('../Data/Output/')
#print("Loading data...")
#for dirID in tqdm(iterable=dataDir, total=len(dataDir)):
#    dr = DataReader('../Data/Output/' + dirID, dirID)
#    totalImageCount += dr.trainDataSize
#    allDatasets.append(dr.dataset)
#print(f"A total of {totalImageCount} have been loaded!")

dataLoader = dl.DatasetLoader('../Data/Output/','')
dataLoader.LoadTrainDatasets()
dataArray = dataLoader.DataSets

bulkDatasetFormatter = df.BulkDatasetFormatter(dataArray, num_classes,batch_size,(image_size,image_size))
tensorDatasets = bulkDatasetFormatter.ProcessData();

#train
train(tensorDatasets,cond_gan,epoch_count)
#cond_gan.fit(dataset, epochs=epoch_count)
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