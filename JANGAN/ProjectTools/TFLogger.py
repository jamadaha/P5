from numpy.core.numeric import normalize_axis_tuple
from ProjectTools import AutoPackageInstaller as ap
import tensorflow as tf
import os

ap.CheckAndInstall("time")
ap.CheckAndInstall("matplotlib")
ap.CheckAndInstall("tensorflow")

class TFLogger:
    __Scope = ''
    __Name = ''
    
    __SummaryWriter = None

    __Letters = []

    def __init__(self, outputDir: str, scope: str, name: str) -> None:
        self.OutputDir = outputDir
        self.__Scope = scope
        self.__Name = name
        self.__SummaryWriter = tf.summary.create_file_writer(os.path.join(outputDir, scope, name))

    def LogNumber(self, number, step):
        with self.__SummaryWriter.as_default():
            with tf.name_scope(self.__Scope):
                tf.summary.scalar(self.__Name, number, step=step)

    def LogGridImages(self, figure, step):
        with self.__SummaryWriter.as_default():
            tf.summary.image("Epoch samples", self.__PlotToImage(figure), step=step)

    def LogConfusionMatrix(self, matrix, step, saveFig, distributionPath):
        import matplotlib.pyplot as plt
        import numpy
        figure = plt.figure(figsize=(8, 8))

        plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.colorbar()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        if not self.__Letters:
            self.__GetLabels(distributionPath)

        tickMarks = numpy.arange(len(self.__Letters))
        plt.xticks(tickMarks, self.__Letters, rotation=45)
        plt.yticks(tickMarks, self.__Letters)

        if saveFig:
            self.SaveMatplot(self.OutputDir, 'Confusion matrix')

        with self.__SummaryWriter.as_default():
            tf.summary.image("Predictions", self.__PlotToImage(figure), step=step)

    #https://www.tensorflow.org/tensorboard/image_summaries
    def __PlotToImage(self, figure):
        import io
        import matplotlib.pyplot as plt
        """Converts the matplotlib plot specified by 'figure' to a PNG image and
        returns it. The supplied figure is closed and inaccessible after this call."""
        # Save the plot to a PNG in memory.
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(figure)
        buf.seek(0)
        # Convert PNG buffer to TF image
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        # Add the batch dimension
        image = tf.expand_dims(image, 0)
        return image

    def SaveMatplot(self, basePath, id):
        import matplotlib.pyplot as plt

        if not os.path.isdir(basePath):
            os.makedirs(basePath)

        plt.savefig(
            fname=basePath + str(id) + '.png',
            format='png'
        )
    
    def __GetLabels(self, distributionPath):
        if not distributionPath:
            return
        import csv
        with open(distributionPath, "r") as file:
            dataReader = csv.reader(file)
            for row in dataReader:
                self.__Letters.append(str(row[0]))
