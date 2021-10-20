import tensorflow as tf
from tensorflow.keras import layers
from multipledispatch import dispatch

IMAGE_RESOLUTION = (28, 28)

class Generator(object):
    """description of class"""

    def __init__(self, *args, **kwargs):
        self.model = self._MakeGeneratorModel()
        return super().__init__(*args, **kwargs)

    def Train(self, labels):
        self.model(noise, training=True)

    @dispatch(object)
    def Generate(self, noise):
        return self.model(noise, training=False)
    
    @dispatch()
    def Generate(self):
        return self.Generate(tf.random.normal([1, 100]))

    def _MakeGeneratorModel(self):
        model = tf.keras.Sequential()
        model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Reshape((7, 7, 256)))
        # Note: None is the batch size
        assert model.output_shape == (None, 7, 7, 256)

        model.add(layers.Conv2DTranspose(
            128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
        assert model.output_shape == (None, 7, 7, 128)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(
            64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, 14, 14, 64)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2),
                  padding='same', use_bias=False, activation='tanh'))
        assert model.output_shape == (
            None, IMAGE_RESOLUTION[0], IMAGE_RESOLUTION[1], 1)

        return model


