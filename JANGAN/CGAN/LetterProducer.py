from ProjectTools import AutoPackageInstaller as ap

ap.CheckAndInstall("tensorflow")
ap.CheckAndInstall("imageio")

import tensorflow as tf
import imageio
import os

class LetterProducer():
    OutputPath = ""
    TrainedGenerator = None
    NumberOfClasses = 0
    LatentDimension = 0
    ImageCountToProduce = 0
    MaxImageBatch = 500

    def __init__(self, outputPath, trainedGenerator, numberOfClasses, latentDimension, imageCountToProduce):
        self.OutputPath = outputPath
        self.TrainedGenerator = trainedGenerator
        self.NumberOfClasses = numberOfClasses
        self.LatentDimension = latentDimension
        self.ImageCountToProduce = imageCountToProduce

    def GenerateLetter(self, classID, imageCountToProduce):
        returnImages = None
        if imageCountToProduce > self.MaxImageBatch:
            returnImages = self.GenerateLetterBatch(classID, self.MaxImageBatch)
            imageCountToProduce -= self.MaxImageBatch
            while imageCountToProduce > 0:
                if imageCountToProduce > self.MaxImageBatch:
                    returnImages = tf.concat([returnImages, (self.GenerateLetterBatch(classID, self.MaxImageBatch))], 0)
                    imageCountToProduce -= self.MaxImageBatch
                else:
                    returnImages = tf.concat([returnImages, (self.GenerateLetterBatch(classID, imageCountToProduce))], 0)
                    imageCountToProduce -= self.MaxImageBatch
        else:
            returnImages = self.GenerateLetterBatch(classID, imageCountToProduce)
        return returnImages

    def GenerateLetterBatch(self, classID, imageCountToProduce):
        # Sample noise for the interpolation.
        interpolation_noise = tf.random.normal(shape=(1, self.LatentDimension))
        for index in range(imageCountToProduce - 1):
            interpolation_noise = tf.concat([interpolation_noise, tf.random.normal(shape=(1, self.LatentDimension))], 0)
        interpolation_noise = tf.reshape(interpolation_noise, (imageCountToProduce, self.LatentDimension))

        label = tf.keras.utils.to_categorical([classID], self.NumberOfClasses)
        label = tf.cast(label, tf.float32)

        # Calculate the interpolation vector between the two labels.
        percent_second_label = tf.linspace(0, 1, imageCountToProduce)[:, None]
        percent_second_label = tf.cast(percent_second_label, tf.float32)
        interpolation_labels = (
            label * (1 - percent_second_label) + label * percent_second_label
        )

        # Combine the noise and the labels and run inference with the generator.
        noise_and_labels = tf.concat([interpolation_noise, interpolation_labels], 1)
        fake = self.TrainedGenerator(noise_and_labels, training=False)
        return fake
        
    def SaveImagesAsGif(self, images):
        images *= 255.0

        #Generate Gif
        imageio.mimsave('out.gif', images)

    def SaveImages(self, basePath, id, images):
        path = basePath + str(id) + '/'

        if not os.path.isdir(path):
            os.makedirs(path)

        # Save images
        index = 0
        for image in images:
            tf.keras.utils.save_img(
                path + str(index) + ".png".format(image), image
            )
            index += 1

    def SaveMatplot(self, basePath, id):
        import matplotlib.pyplot as plt

        if not os.path.isdir(basePath):
            os.makedirs(basePath)

        plt.savefig(
            fname=basePath + str(id) + '.png',
            format='png'
        )

    def GetSampleLetters(self):
        imageArray = []

        for i in range(self.NumberOfClasses):
            imageArray.append(self.GenerateLetter(i, 1))

        return imageArray
        

    def ProduceLetters(self):
        from tqdm import tqdm

        for i in tqdm(range(self.NumberOfClasses), desc='Producing images'):
            images = self.GenerateLetter(i, self.ImageCountToProduce)
            self.SaveImages(
                self.OutputPath, 
                i, 
                images)

    def ProduceGridLetters(self, id):
        import matplotlib.pyplot as plt
        import math
        import numpy

        images = self.GetSampleLetters()
        imageArray = numpy.reshape(images, (len(images), 28, 28, 1))

        figure = plt.figure(figsize=(8, 8))
        gridSize = math.ceil(math.sqrt(len(imageArray)))
        for i in range(len(imageArray)):
            plt.subplot(gridSize, gridSize, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.tight_layout()
            plt.imshow(imageArray[i], cmap=plt.cm.binary)

        self.SaveMatplot(self.OutputPath, id)

        return figure