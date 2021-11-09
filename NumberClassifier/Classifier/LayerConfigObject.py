from tensorflow.keras import layers
class LayerConfigObject(object):
    """description of class"""

    def __init__(self):
        self.layers = [
                    layers.Rescaling(1./255),
                    layers.Conv2D(32, 3, activation='relu'),
                    layers.MaxPooling2D(),
                    layers.Conv2D(32, 3, activation='relu'),
                    layers.MaxPooling2D(),
                    layers.Conv2D(32, 3, activation='relu'),
                    layers.MaxPooling2D(),
                    layers.Flatten(),
                    layers.Dense(128, activation='relu')]

    def add_dense_layer(self, num: int):
        self.layers.append(layers.Dense(num))
