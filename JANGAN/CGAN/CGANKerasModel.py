from ProjectTools import AutoPackageInstaller as ap

ap.CheckAndInstall("tensorflow")

import tensorflow as tf

class CGANKerasModel(tf.keras.Model):
    ImageSize = 0
    NumberOfClasses = 0
    TrackModeCollapse = False
    Discriminator = None
    Generator = None
    LatentDimension = 0
    DiscriminatorLossTracker = None
    GeneratorLossTracker = None
    ModeCollapseTracker = None

    DiscriminatorOptimizer = None
    GeneratorOptimizer = None

    LossFunction = None

    def __init__(self, discriminator, generator, latentDimension, imageSize, numberOfClasses, trackModeCollapse):
        super(CGANKerasModel, self).__init__()
        self.Discriminator = discriminator
        self.Generator = generator
        self.LatentDimension = latentDimension
        self.GeneratorLossTracker = tf.keras.metrics.Mean(name="generator_loss")
        self.DiscriminatorLossTracker = tf.keras.metrics.Mean(name="discriminator_loss")
        self.ModeCollapseTracker = tf.keras.metrics.Mean(name="mode_collapse_tracker")
        self.ImageSize = imageSize
        self.NumberOfClasses = numberOfClasses
        self.TrackModeCollapse = trackModeCollapse

    @property
    def metrics(self):
        return [self.GeneratorLossTracker, self.DiscriminatorLossTracker, self.ModeCollapseTracker]

    def compile(self, discriminatorOptimizer, generatorOptimizer, lossFunction):
        super(CGANKerasModel, self).compile()
        self.DiscriminatorOptimizer = discriminatorOptimizer
        self.GeneratorOptimizer = generatorOptimizer
        self.LossFunction = lossFunction

    @tf.function
    def train_step(self, data, returnLoss):
         # Unpack the data.
        realImages, oneHotLabels = data

        # Add dummy dimensions to the labels so that they can be concatenated with
        # the images. This is for the discriminator.
        imageOneHotLabels = oneHotLabels[:, :, None, None]
        imageOneHotLabels = tf.repeat(
            imageOneHotLabels, repeats=[self.ImageSize * self.ImageSize]
        )
        imageOneHotLabels = tf.reshape(
            imageOneHotLabels, (-1, self.ImageSize, self.ImageSize, self.NumberOfClasses)
        )

        # Sample random points in the latent space and concatenate the labels.
        # This is for the generator.
        batchSize = tf.shape(realImages)[0]
        randomLatentVectors = tf.random.normal(shape=(batchSize, self.LatentDimension))
        randomVectorLabels = tf.concat(
            [randomLatentVectors, oneHotLabels], axis=1
        )

        # Decode the noise (guided by labels) to fake images.
        generatedImages = self.Generator(randomVectorLabels, training=True)

        if self.TrackModeCollapse == True:
            if returnLoss == True:
                modeLoss = tf.reduce_sum(tf.image.total_variation(generatedImages))

        # Combine them with real images. Note that we are concatenating the labels
        # with these images here.
        fakeImagesAndLabels = tf.concat([generatedImages, imageOneHotLabels], -1)
        realImagesAndLabels = tf.concat([realImages, imageOneHotLabels], -1)
        combinedImages = tf.concat(
            [fakeImagesAndLabels, realImagesAndLabels], axis=0
        )

        # Assemble labels discriminating real from fake images.
        labels = tf.concat(
            [tf.ones((batchSize, 1)), tf.zeros((batchSize, 1))], axis=0
        )

        # Train the discriminator.
        with tf.GradientTape() as tape:
            predictions = self.Discriminator(combinedImages, training=True)
            discLoss = self.LossFunction(labels, predictions)
        grads = tape.gradient(discLoss, self.Discriminator.trainable_weights)
        self.DiscriminatorOptimizer.apply_gradients(
            zip(grads, self.Discriminator.trainable_weights)
        )

        # Sample random points in the latent space.
        randomLatentVectors = tf.random.normal(shape=(batchSize, self.LatentDimension))
        randomVectorLabels = tf.concat(
            [randomLatentVectors, oneHotLabels], axis=1
        )

        # Assemble labels that say "all real images".
        fakeLabels = tf.zeros((batchSize, 1))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tf.GradientTape() as tape:
            fakeImages = self.Generator(randomVectorLabels, training=True)
            fakeImagesAndLabels = tf.concat([fakeImages, imageOneHotLabels], -1)
            predictions = self.Discriminator(fakeImagesAndLabels, training=True)
            genLoss = self.LossFunction(fakeLabels, predictions)
        grads = tape.gradient(genLoss, self.Generator.trainable_weights)
        self.GeneratorOptimizer.apply_gradients(zip(grads, self.Generator.trainable_weights))

        # Monitor loss.
        if returnLoss == True:
            self.GeneratorLossTracker.update_state(genLoss)
            self.DiscriminatorLossTracker.update_state(discLoss)
            if self.TrackModeCollapse == True:
                self.ModeCollapseTracker.update_state(modeLoss)
            return {
                "g_loss": self.GeneratorLossTracker.result(),
                "d_loss": self.DiscriminatorLossTracker.result(),
                "mode_collapse_loss": self.ModeCollapseTracker.result()
            }
