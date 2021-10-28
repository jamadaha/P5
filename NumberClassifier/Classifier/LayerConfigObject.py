class LayerConfigObject(object):
    """description of class"""

    def __init__(self):
        self.layers = [
                    tensorflow.keras.layers.Rescaling(1./255),
                    tensorflow.keras.layers.Conv2D(32, 3, activation='relu'),
                    tensorflow.keras.layers.MaxPooling2D(),
                    tensorflow.keras.layers.Conv2D(32, 3, activation='relu'),
                    tensorflow.keras.layers.MaxPooling2D(),
                    tensorflow.keras.layers.Conv2D(32, 3, activation='relu'),
                    tensorflow.keras.layers.MaxPooling2D(),
                    tensorflow.keras.layers.Flatten(),
                    tensorflow.keras.layers.Dense(128, activation='relu'),
                    tensorflow.keras.layers.Dense(num_classes)]
