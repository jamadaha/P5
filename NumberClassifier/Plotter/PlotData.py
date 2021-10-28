from multipledispatch import dispatch

class PlotData(object):
    """description of class"""

    @dispatch(list)
    def __init__(self, epoch_range: list):
        self.epoch_range = epoch_range
        self.data = dict()

    @dispatch(int)
    def __init__(self, epoch_range: int):
        self.epoch_range = range(epoch_range)

    def add_dataset(self, name: str, dataset: list):
        self.data[name] = dataset