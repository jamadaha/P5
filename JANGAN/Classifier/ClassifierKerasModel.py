from ProjectTools import AutoPackageInstaller as ap

ap.CheckAndInstall("tensorflow")

import tensorflow as tf

class ClassifierKerasModel(tf.keras.Model):
    ImageSize = 0
    NumberOfClasses = 0
    Classifier = None
    LossTracker = None
    AccuracyTracker = None
    ImageSize = 0
    NumberOfClasses = 0

    Optimizer = None

    LossFunction = None

    def __init__(self, classifier, imageSize, numberOfClasses):
        super(ClassifierKerasModel, self).__init__()
        self.Classifier = classifier
        self.LossTracker = tf.keras.metrics.Mean(name="classifier_loss")
        self.AccuracyTracker = tf.keras.metrics.CategoricalAccuracy(name="classifier_accuracy")
        self.ImageSize = imageSize
        self.NumberOfClasses = numberOfClasses

    @property
    def metrics(self):
        return [self.LossTracker, self.AccuracyTracker]

    def compile(self, optimizer, lossFunction):
        super(ClassifierKerasModel, self).compile()
        self.Optimizer = optimizer
        self.LossFunction = lossFunction

    @tf.function
    def train_step(self, data, returnLoss):
         # Unpack the data.
        realImages, oneHotLabels = data

        # Train the classifier.
        with tf.GradientTape() as tape:
            predictions = self.Classifier(realImages, training=True)
            classifierLoss = self.LossFunction(oneHotLabels, predictions)
        grads = tape.gradient(classifierLoss, self.Classifier.trainable_weights)
        self.optimizer.apply_gradients(
            zip(grads, self.Classifier.trainable_weights)
        )
        
        # Monitor loss.
        if returnLoss == True:
            self.LossTracker.update_state(classifierLoss)
            return {
                "c_loss": self.LossTracker.result()
            }

    @tf.function
    def test_step(self, data, returnAccuracy):
        realImages, realLabels = data

        # Combine the real images and the base tensor from before, and make a prediction on it
        predictions = self.Classifier(realImages, training=False)

        # Update loss for this batch
        self.AccuracyTracker.update_state(realLabels, predictions)

        if returnAccuracy == True:
            return {
                "classifier_accuracy": self.AccuracyTracker.result()
            }
