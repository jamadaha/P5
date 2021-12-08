from ProjectTools import AutoPackageInstaller as ap

ap.CheckAndInstall("tensorflow")

import tensorflow as tf

class ConditionalGAN(tf.keras.Model):
    ImageSize = 0
    NumberOfClasses = 0
    LatestImage = None
    TrackModeCollapse = False

    def __init__(self, discriminator, generator, latentDimension, imageSize, numberOfClasses, trackModeCollapse):
        super(ConditionalGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latentDimension
        self.gen_loss_tracker = tf.keras.metrics.Mean(name="generator_loss")
        self.disc_loss_tracker = tf.keras.metrics.Mean(name="discriminator_loss")
        self.mode_collapse_tracker = tf.keras.metrics.Mean(name="mode_collapse_tracker")
        self.ImageSize = imageSize
        self.NumberOfClasses = numberOfClasses
        self.TrackModeCollapse = trackModeCollapse

    @property
    def metrics(self):
        return [self.gen_loss_tracker, self.disc_loss_tracker, self.mode_collapse_tracker]

    def compile(self, d_optimizer, g_optimizer, loss_fn, mode_collapse_loss_fn):
        super(ConditionalGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn
        self.mode_collapse_loss_fn = mode_collapse_loss_fn

    @tf.function
    def train_step(self, data, returnLoss):
         # Unpack the data.
        real_images, one_hot_labels = data

        # Add dummy dimensions to the labels so that they can be concatenated with
        # the images. This is for the discriminator.
        image_one_hot_labels = one_hot_labels[:, :, None, None]
        image_one_hot_labels = tf.repeat(
            image_one_hot_labels, repeats=[self.ImageSize * self.ImageSize]
        )
        image_one_hot_labels = tf.reshape(
            image_one_hot_labels, (-1, self.ImageSize, self.ImageSize, self.NumberOfClasses)
        )

        # Sample random points in the latent space and concatenate the labels.
        # This is for the generator.
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        random_vector_labels = tf.concat(
            [random_latent_vectors, one_hot_labels], axis=1
        )

        # Decode the noise (guided by labels) to fake images.
        generated_images = self.generator(random_vector_labels, training=True)

        # Combine them with real images. Note that we are concatenating the labels
        # with these images here.
        fake_image_and_labels = tf.concat([generated_images, image_one_hot_labels], -1)
        real_image_and_labels = tf.concat([real_images, image_one_hot_labels], -1)
        combined_images = tf.concat(
            [fake_image_and_labels, real_image_and_labels], axis=0
        )

        # Assemble labels discriminating real from fake images.
        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )

        # Train the discriminator.
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images, training=True)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        # Sample random points in the latent space.
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        random_vector_labels = tf.concat(
            [random_latent_vectors, one_hot_labels], axis=1
        )

        # Assemble labels that say "all real images".
        misleading_labels = tf.zeros((batch_size, 1))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tf.GradientTape() as tape:
            fake_images = self.generator(random_vector_labels, training=True)
            fake_image_and_labels = tf.concat([fake_images, image_one_hot_labels], -1)
            predictions = self.discriminator(fake_image_and_labels, training=True)
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        if self.TrackModeCollapse == True:
            random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
            random_vector_labels = tf.concat(
                [random_latent_vectors, one_hot_labels], axis=1
            )

            generated_images = self.generator(random_vector_labels, training=False)
        
            modeLoss = tf.reduce_sum(tf.image.total_variation(generated_images))

        # Monitor loss.
        if returnLoss == True:
            self.gen_loss_tracker.update_state(g_loss)
            self.disc_loss_tracker.update_state(d_loss)
            self.mode_collapse_tracker.update_state(modeLoss)
            return {
                "g_loss": self.gen_loss_tracker.result(),
                "d_loss": self.disc_loss_tracker.result(),
                "mode_collapse_loss": self.mode_collapse_tracker.result()
            }
