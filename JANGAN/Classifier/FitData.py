import tensorflow
class FitData(object):
    """Datastructure for holding a training and evaluation dataset for fitting models."""
    
    def __init__(self, td: tensorflow.data.Dataset, vd: tensorflow.data.Dataset):
        self.train_data = td
        self.val_data = vd
        self.labels: []
        self.num_classes = len(td.class_names)

    def get_train_data(self):
        return self.train_data

    def set_train_data(self, data: tensorflow.data.Dataset):
        self.train_data = data

    def get_val_data(self):
        return self.val_data
    
    def set_val_data(self, data: tensorflow.data.Dataset):
        self.val_data = data

