from tensorflow import data
class FittingData(object):
    """Datastructure for holding a training and evaluation dataset for fitting models."""

    def __init__(self, train_data: data.Dataset, val_data: data.Dataset):
        self.train_data = train_data
        self.val_data = val_data
        self.__num_classes = len(train_data.class_names)

    def get_train_data(self):
        return self.train_data

    def set_train_data(self, data: data.Dataset):
        self.train_data = data

    def get_val_data(self):
        return self.val_data
    
    def set_val_data(self, data: data.Dataset):
        self.val_data = data




