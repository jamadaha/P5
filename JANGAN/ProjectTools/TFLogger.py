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
