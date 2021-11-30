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

    def __init__(self, outputDir: str, scope: str, name: str) -> None:
        self.OutputDir = outputDir
        self.Scope = scope
        self.Name = name
        self.__SummaryWriter = tf.summary.create_file_writer(os.path.join(outputDir, scope, name))

    def LogNumber(self, number, step):
        with self.__SummaryWriter.as_default():
            with tf.name_scope(self.__Scope):
                tf.summary.scalar(self.__Name, number, step=step)

    def LogGridImages(self, images, step):
        import numpy
        imageArray = numpy.reshape(images, (len(images), 28, 28, 1))

        figure = self.__FigGrid(imageArray)

        with self.__SummaryWriter.as_default():
            tf.summary.image("Epoch samples", self.__PlotToImage(figure), max_outputs=len(imageArray), step=step)

    def __FigGrid(self, images):
        import matplotlib.pyplot as plt
        import math
        figure = plt.figure(figsize=(10, 10))
        gridSize = math.ceil(math.sqrt(len(images)))
        for i in range(len(images)):
            plt.subplot(gridSize, gridSize, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.tight_layout()
            plt.imshow(images[i], cmap=plt.cm.binary)
        return figure

    #https://www.tensorflow.org/tensorboard/image_summaries
    def __PlotToImage(self, figure):
        import io
        import matplotlib.pyplot as plt
        """Converts the matplotlib plot specified by 'figure' to a PNG image and
        returns it. The supplied figure is closed and inaccessible after this call."""
        # Save the plot to a PNG in memory.
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        # Closing the figure prevents it from being displayed directly inside
        # the notebook.
        plt.close(figure)
        buf.seek(0)
        # Convert PNG buffer to TF image
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        # Add the batch dimension
        image = tf.expand_dims(image, 0)
        return image
