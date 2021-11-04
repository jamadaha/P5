from ProjectTools import AutoPackageInstaller as ap

ap.CheckAndInstall("tensorflow")
ap.CheckAndInstall("imageio")
ap.CheckAndInstall("pillow")

from tensorflow import keras
import tensorflow as tf
import imageio
import os
import matplotlib.pyplot as plt

class LetterProducer():
    OutputPath = ""
    TrainedGenerator = None
    NumberOfClasses = 0
    LatentDimension = 0

    def __init__(self, outputPath, trainedGenerator, numberOfClasses, latentDimension):
        self.OutputPath = outputPath
        self.TrainedGenerator = trainedGenerator
        self.NumberOfClasses = numberOfClasses
        self.LatentDimension = latentDimension

    def GenerateLetter(self, classID, imageCountToProduce):
        # Sample noise for the interpolation.
        interpolation_noise = tf.random.normal(shape=(1, self.LatentDimension))
        interpolation_noise = tf.repeat(interpolation_noise, repeats=imageCountToProduce)
        interpolation_noise = tf.reshape(interpolation_noise, (imageCountToProduce, self.LatentDimension))

        first_label = keras.utils.to_categorical([classID], self.NumberOfClasses)
        second_label = keras.utils.to_categorical([classID], self.NumberOfClasses)
        first_label = tf.cast(first_label, tf.float32)
        second_label = tf.cast(second_label, tf.float32)

        # Calculate the interpolation vector between the two labels.
        percent_second_label = tf.linspace(0, 1, imageCountToProduce)[:, None]
        percent_second_label = tf.cast(percent_second_label, tf.float32)
        interpolation_labels = (
            first_label * (1 - percent_second_label) + second_label * percent_second_label
        )

        # Combine the noise and the labels and run inference with the generator.
        noise_and_labels = tf.concat([interpolation_noise, interpolation_labels], 1)
        fake = self.TrainedGenerator.predict(noise_and_labels)
        return fake
        
    def SaveImagesAsGif(self, images):
        images *= 255.0

        #Generate Gif
        imageio.mimsave('out.gif', images)

    def SaveImages(self, id, images):
        path = self.OutputPath + str(id) + '/'

        if not os.path.isdir(path):
            os.makedirs(path)

        # Save images
        index = 0
        for image in images:
            plt.imshow(image[:, :, 0], cmap='gray')
            plt.axis('off')
            plt.savefig(path + str(index) + ".png".format(image), bbox_inches='tight', pad_inches=0)
            index += 1
