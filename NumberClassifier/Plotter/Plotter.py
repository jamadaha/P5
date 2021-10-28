from tensorflow import data
import matplotlib.pyplot as plt
from PlotData import PlotData

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

    def __plot_lines__(self, data: PlotData):
        for i in data.data:
            plt.plot(data.range, data.data[i], label=i)

    def __plot_subplot__(self, name:str, data: PlotData):
        plt.subplot(1,2,1)
        self.__plot_lines__(data)
        plt.legend(loc='lower right')
        plt.title(name)

    def plot_graph(self, title: str, data: PlotData):
        self.__plot_subplot__(data)
        plt.show()
    

