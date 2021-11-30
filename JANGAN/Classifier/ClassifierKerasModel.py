from ProjectTools import AutoPackageInstaller as ap

ap.CheckAndInstall("tensorflow")

import tensorflow as tf

class ClassifierModel(tf.keras.Model):
    ImageSize = 0
    NumberOfClasses = 0

    def __init__(self, classifier, imageSize, numberOfClasses, accuracyThreshold):
        super(ClassifierModel, self).__init__()
        self.classifier = classifier
        self.loss_tracker = tf.keras.metrics.Mean(name="classifier_loss")
        self.accuracy = tf.keras.metrics.BinaryAccuracy(name="classifier_accuracy", threshold=accuracyThreshold)
        self.ImageSize = imageSize
        self.NumberOfClasses = numberOfClasses

    @property
    def metrics(self):
        return [self.loss_tracker, self.accuracy]

    def compile(self, optimizer, loss_fn):
        super(ClassifierModel, self).compile()
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    @tf.function
    def train_step(self, data, returnLoss):
         # Unpack the data.
        real_images, one_hot_labels = data

        # Make the "correct" labels, consisting of a large array with '1's
        batch_size = tf.shape(real_images)[0]
        correct_labels = tf.ones((batch_size, 1))

        # Train the classifier.
        with tf.GradientTape() as tape:
            predictions = self.classifier(real_images, training=True)
            c_loss = self.loss_fn(one_hot_labels, predictions)
        grads = tape.gradient(c_loss, self.classifier.trainable_weights)
        self.optimizer.apply_gradients(
            zip(grads, self.classifier.trainable_weights)
        )
        
        # Monitor loss.
        if returnLoss == True:
            self.loss_tracker.update_state(c_loss)
            return {
                "c_loss": self.loss_tracker.result()
            }

    @tf.function
    def test_step(self, data, returnAccuracy):
        real_images, real_labels = data

        # Combine the real images and the base tensor from before, and make a prediction on it
        predictions = self.classifier(real_images, training=False)

        # Update loss for this batch
        self.accuracy.update_state(real_labels, predictions)

        if returnAccuracy == True:
            return {
                "classifier_accuracy": self.accuracy.result()
            }
