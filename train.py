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
        GaussianNoise(0.115, input_shape=(config.NOISE_DIM,)),  # Add noise to input
        Dense(config.GEN_COMPLEXITY, activation='relu', input_shape=(100,)),  # 100-dimensional noise
        BatchNormalization(),
        Dense(config.GEN_COMPLEXITY, activation='LeakyReLU'), # add an additional layer
        Dropout(config.DROPOUT_RATE),                               # add dropout
        MiniBatchDiscrimination(num_kernels=30, kernel_dim=3),
        Dense(784, activation='sigmoid'),           # Reshape to 28x28 image
        Reshape((28, 28))
    ])
    return model

def build_discriminator():
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(config.DISC_COMPLEXITY, activation='LeakyReLU'),
        Dropout(config.DROPOUT_RATE),  # Add dropout
        MiniBatchDiscrimination(num_kernels=50, kernel_dim=5),
        Dense(1, activation='sigmoid')
    ])
    return model

def log_to_tensorboard(writer, name, text, step):
    with writer.as_default():
        tf.summary.text(name, data=text, step=step)

def train(generator, gen_opt, discriminator, disc_opt, dataset, start_epoch, epochs, writer):
    with writer.as_default():
        for epoch in range(start_epoch, epochs):
            start_time = time.time()
            for image_batch in dataset:
                noise = tf.random.normal([config.BATCH_SIZE, 100])

                with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                    generated_images = generator(noise, training=True)

                    real_output = discriminator(image_batch, training=True)
                    fake_output = discriminator(generated_images, training=True)
                    
                    # Adding noise to labels
                    real_label_noise = tf.random.uniform(shape=tf.shape(real_output), minval=0.0, maxval=0.3)
                    fake_label_noise = tf.random.uniform(shape=tf.shape(fake_output), minval=0.0, maxval=0.3)

                    gen_loss = generator_loss(fake_output)
                    disc_loss = discriminator_loss(real_output, fake_output, real_label_noise, fake_label_noise)

                gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
                gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

                gen_opt.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
                disc_opt.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

            if (epoch % config.LOGGING_INTERVAL) == 0 and epoch != start_epoch:
                saveNload.save_model_weights(generator, discriminator, epoch, writer)

            # Log the time it takes for each epoch
            duration = time.time() - start_time
            print(f'Epoch {epoch+1}/{config.EPOCHS} completed in {duration:.2f} seconds')
            name = "Epoch Status"
            text = "Epoch " + str(epoch+1) +"/" + config.EPOCHS + " completed in " + duration + " seconds"
            log_to_tensorboard(writer, name, text, epoch)

            # Log the losses to TensorBoard
            with writer.as_default():
                tf.summary.scalar('Generator Loss', gen_loss, step=epoch)
                tf.summary.scalar('Discriminator Loss', disc_loss, step=epoch)
                writer.flush()
            if (epoch % 5) == 0:
                saveNload.generate_and_save_images(generator, epoch, seed, writer)

    # Generate after the final epoch
    saveNload.generate_and_save_images(generator, epochs, seed, writer)
    exit

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)

"""def discriminator_loss(real_labels, fake_labels, real_output, fake_output):
    real_loss = cross_entropy(real_labels * config.DISC_CONFIDENCE, real_output)
    fake_loss = cross_entropy(fake_labels, fake_output)
    total_loss = real_loss + fake_loss
    return total_loss"""

def discriminator_loss(real_output, fake_output, real_label_noise, fake_label_noise):
    noisy_real_labels = tf.ones_like(real_output) - real_label_noise
    noisy_fake_labels = tf.zeros_like(fake_output) + fake_label_noise

    real_loss = cross_entropy(noisy_real_labels * config.DISC_CONFIDENCE, real_output)
    fake_loss = cross_entropy(noisy_fake_labels, fake_output)
    
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)