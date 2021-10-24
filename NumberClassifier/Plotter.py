from tensorflow import data
import matplotlib.pyplot as plt

class Plotter(object):
    """Utility class for plotting data"""

    def plot_dataset_slice(self, dataset: data.Dataset, n: int):
        plt. figure(figsize(10,10))

        for images, lavels in train_ds.take(1):
            for i in range(n):
                ax = plt.subplot(3,3,i+1)
                plt.imshow(images[i].numpy().astype("uint8"))
                plt.title(class_names[labels[i]])
                plt.axis("off")

        plt.show()

    def __plot_linegraph__(self, epochs, name:str):

        epochs_range = range(epochs)
        plt.subplot(1,2,1)
        plt.plot(epochs_range, acc, label=name)
        plt.legend(loc='lower right')
        plt.title(name)
