import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Reshape, GaussianNoise, BatchNormalization, Dropout, Layer
from tensorflow.keras import Sequential
import config, saveNload, time

seed = tf.random.normal([config.NUM_EXAMPLES_TO_GEN, config.NOISE_DIM])

class MiniBatchDiscrimination(Layer):
    def __init__(self, num_kernels, kernel_dim, **kwargs):
        super(MiniBatchDiscrimination, self).__init__(**kwargs)
        self.num_kernels = num_kernels
        self.kernel_dim = kernel_dim

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name='kernel',
            shape=(input_shape[-1], self.num_kernels * self.kernel_dim),
            initializer='glorot_uniform',
            trainable=True)

    def call(self, input):
        activation = tf.matmul(input, self.kernel)
        activation = tf.reshape(activation, (-1, self.num_kernels, self.kernel_dim))
        diff = tf.expand_dims(activation, 3) - tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0)
        l1_norm = tf.reduce_sum(tf.abs(diff), axis=2)
        mb_feats = tf.reduce_sum(tf.exp(-l1_norm), axis=-1)
        return tf.concat([input, mb_feats], axis=1)

    def get_config(self):
        config = super(MiniBatchDiscrimination, self).get_config()
        config.update({
            'num_kernels': self.num_kernels,
            'kernel_dim': self.kernel_dim
        })
        return config


def build_generator():
    model = Sequential([
        GaussianNoise(0.115, input_shape=(config.NOISE_DIM,)),      # Add noise to input
        Dense(config.GEN_COMPLEXITY, activation='relu', input_shape=(100,)),  # 100-dimensional noise
        BatchNormalization(),
        Dropout(config.DROPOUT_RATE),                               # add dropout
        Dense(config.GEN_COMPLEXITY, activation='LeakyReLU'),       # add an additional layer
        Dropout(config.DROPOUT_RATE),                               # add dropout
        MiniBatchDiscrimination(30, 3),                             # mini batch discrimination
        Dense(config.GEN_COMPLEXITY, activation='LeakyReLU'),       # add an additional layer
        Dropout(config.DROPOUT_RATE),                               # add dropout
        Dense(784, activation='sigmoid'),
        Reshape((28, 28))                                           # Reshape to 28x28 image
    ])
    return model

def build_discriminator():
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(config.DISC_COMPLEXITY, activation='LeakyReLU'),
        Dropout(config.DROPOUT_RATE),                               # Add dropout
        MiniBatchDiscrimination(50, 5),                             # mini batch discrimination
        Dense(config.DISC_COMPLEXITY, activation='LeakyReLU'),
        Dropout(config.DROPOUT_RATE),                               # Add 2nd dropout
        Dense(1, activation='sigmoid')
    ])
    return model

def log_to_tensorboard(writer, name, text, step):
    with writer.as_default():
        tf.summary.text(name, data=text, step=step)

def train(generator, discriminator, dataset, start_epoch, writer):
    # set up optimizers
    generator_optimizer = tf.keras.optimizers.Adam(config.GEN_LEARN_RATE)
    discriminator_optimizer = tf.keras.optimizers.Adam(config.DISC_LEARN_RATE)
    # Setup the binary cross entropy loss
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    
    # Define loss functions
    def discriminator_loss(real_output, fake_output):
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(fake_output):
        return cross_entropy(tf.ones_like(fake_output), fake_output)
    
    # Average metrics trackers
    avg_gen_loss_tracker = tf.keras.metrics.Mean(name='avg_gen_loss')
    avg_disc_loss_tracker = tf.keras.metrics.Mean(name='avg_disc_loss')
    avg_real_accuracy_tracker = tf.keras.metrics.Mean(name='avg_real_accuracy')
    avg_fake_accuracy_tracker = tf.keras.metrics.Mean(name='avg_fake_accuracy')
    
    with writer.as_default():
        for epoch in range(start_epoch, config.EPOCHS):
            start_time = time.time()
            for image_batch in dataset:
                noise = tf.random.normal([config.BATCH_SIZE, config.NOISE_DIM])
                
                # Train the discriminator with real and fake images
                with tf.GradientTape() as disc_tape:
                    real_output = discriminator(image_batch, training=True)
                    generated_images = generator(noise, training=False)
                    fake_output = discriminator(generated_images, training=True)
                    disc_loss = discriminator_loss(real_output, fake_output)
                    
                gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
                discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
                # Calculate and log discriminator accuracy
                real_accuracy = tf.reduce_mean(tf.cast(tf.less(real_output, 0.5), tf.float32))
                fake_accuracy = tf.reduce_mean(tf.cast(tf.greater(fake_output, 0.5), tf.float32))
                
                # Train the generator through the combined model
                with tf.GradientTape() as gen_tape:
                    generated_images = generator(noise, training=True)
                    fake_output = discriminator(generated_images, training=False)
                    gen_loss = generator_loss(fake_output)

                gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
                generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
                
                # Track average losses and accuracies
                avg_gen_loss_tracker.update_state(gen_loss)
                avg_disc_loss_tracker.update_state(disc_loss)
                avg_real_accuracy_tracker.update_state(real_accuracy)
                avg_fake_accuracy_tracker.update_state(fake_accuracy)

            if (epoch % config.CHECKPOINT_INTERVAL) == 0 and epoch != start_epoch:
                saveNload.save_model_weights(generator, discriminator, epoch, writer)

            # Log the time it takes for each epoch
            duration = time.time() - start_time
            print(f'Epoch {epoch+1}/{config.EPOCHS} completed in {duration:.2f} seconds')
            name = "Epoch Status"
            text = "Epoch " + str(epoch+1) +"/" + str(config.EPOCHS) + " completed in " + str(duration) + " seconds"
            log_to_tensorboard(writer, name, text, epoch)
            
            # Log the losses and accuracies to TensorBoard
            # Log the average metrics for the epoch
            with writer.as_default():
                tf.summary.scalar('Average Generator Loss', avg_gen_loss_tracker.result(), step=epoch)
                tf.summary.scalar('Average Discriminator Loss', avg_disc_loss_tracker.result(), step=epoch)
                tf.summary.scalar('Average Real Accuracy', avg_real_accuracy_tracker.result(), step=epoch)
                tf.summary.scalar('Average Fake Accuracy', avg_fake_accuracy_tracker.result(), step=epoch)
                writer.flush()
            if (epoch % 5) == 0:
                saveNload.generate_and_save_images(generator, epoch, seed, writer)
            # Reset metrics every epoch
            avg_gen_loss_tracker.reset_states()
            avg_disc_loss_tracker.reset_states()
            avg_real_accuracy_tracker.reset_states()
            avg_fake_accuracy_tracker.reset_states()

    # Generate after the final epoch
    saveNload.generate_and_save_images(generator, config.EPOCHS, seed, writer)
    exit