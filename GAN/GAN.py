from numpy.core.records import array
import tensorflow as tf
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time

from IPython import display

from multipledispatch import dispatch

from GAN import Generator
from GAN import Discriminator

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

class GAN(object):
    """description of class"""

    def __init__(self, checkpointDir):
        self.Generator = Generator.Generator()
        self.GeneratorOptimizer = tf.keras.optimizers.Adam(1e-4)
        self.Discriminator = Discriminator.Discriminator()
        self.DiscriminatorOptimizer = tf.keras.optimizers.Adam(1e-4)

        self.checkpointDir = checkpointDir
        self.CheckpointManager = tf.train.Checkpoint(generator_optimizer=self.GeneratorOptimizer,
                                         discriminator_optimizer=self.DiscriminatorOptimizer,
                                         generator=self.Generator.model,
                                         discriminator=self.Discriminator.model)

    #@tf.function
    def Train(self, image, label):

        BATCH_SIZE = 256
        noise_dim = 100
        noise = tf.random.normal([BATCH_SIZE, noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_image = self.Generator.model(noise, training=True)

            real_output = self.Discriminator.model(image, training=True)
            fake_output = self.Discriminator.model(generated_image, training=True)

            gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.Generator.model.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.Discriminator.model.trainable_variables)

        self.GeneratorOptimizer.apply_gradients(zip(gradients_of_generator, self.Generator.model.trainable_variables))
        self.DiscriminatorOptimizer.apply_gradients(zip(gradients_of_discriminator, self.Discriminator.model.trainable_variables))
    
    def SaveCheckpoint(self, name):
        path = os.path.join(self.checkpointDir, name)
        self.CheckpointManager.save(file_prefix=path)
    def LoadCheckpoint(self, name):
        path = os.path.join(self.checkpointDir, name)
        self.CheckpointManager.restore(file_prefix=path)


    @dispatch(object)
    def Generate(self, noise):
        return self.Generator.Generate(noise)

    @dispatch()
    def Generate(self):
        return self.Generator.Generate()

    def Discriminate(self, image, label):
        return self.Discriminator.Discriminate(image, label)






