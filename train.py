import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Reshape, GaussianNoise, BatchNormalization, Dropout, Layer
from tensorflow.keras import Sequential
import config, saveNload, time

seed = tf.random.normal([config.NUM_EXAMPLES_TO_GEN, config.NOISE_DIM])
train_dataset = tf.data.Dataset.from_tensor_slices(saveNload.train_images).shuffle(config.BUFFER_SIZE).batch(config.BATCH_SIZE)

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
        # Dense(gen_complexity/2, activation='relu'), # add an additional layer half as complex
        Dropout(0.3),                               # add dropout
        Dense(784, activation='sigmoid'),           # Reshape to 28x28 image
        Reshape((28, 28))
    ])
    return model

def build_discriminator():
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(config.DISC_COMPLEXITY, activation='LeakyReLU'),
        Dropout(0.4),  # Add dropout
        MiniBatchDiscrimination(num_kernels=50, kernel_dim=5),
        Dense(1, activation='sigmoid')
    ])
    return model

def train(generator, gen_opt, discriminator, disc_opt, start_epoch, writer, lambda_div=config.LAMBDA_DIV, gamma=config.GAMMA):
    with writer.as_default():
        for epoch in range(start_epoch, config.EPOCHS):
            start_time = time.time()
            for input_sentence in train_dataset:
                train_step(generator, discriminator, input_sentence, gen_opt, disc_opt, lambda_div, gamma)
            for image_batch in train_dataset:
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

            if (epoch % 10) == 0 and epoch != start_epoch:
                saveNload.save_model_weights(generator, discriminator, epoch)

            # Log the time it takes for each epoch
            duration = time.time() - start_time
            print(f'Epoch {epoch+1}/{epochs} completed in {duration:.2f} seconds')

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

def pairwise_distances(embeddings, squared=False):
    # Calculate the pairwise distance matrix
    dot_product = tf.matmul(embeddings, tf.transpose(embeddings))
    square_norm = tf.linalg.diag_part(dot_product)
    distances = tf.expand_dims(square_norm, 0) - 2.0 * dot_product + tf.expand_dims(square_norm, 1)
    
    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances = tf.maximum(distances, 0.0)
    if not squared:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = tf.cast(tf.equal(distances, 0.0), tf.float32)
        distances = distances + mask * 1e-16

        distances = tf.sqrt(distances)

        # Correct the epsilon added: set the distances on the mask to be exactly 0.0
        distances = distances * (1.0 - mask)

    return distances

def diversity_loss(embeddings, latent_codes, lambda_div):
    pairwise_emb_distances = pairwise_distances(embeddings)
    pairwise_latent_distances = pairwise_distances(latent_codes)
    
    # Avoid division by zero
    pairwise_latent_distances += 1e-8
    
    ratios = pairwise_emb_distances / pairwise_latent_distances
    div_loss = tf.maximum(lambda_div - ratios, 0)
    div_loss = tf.reduce_mean(div_loss)
    return div_loss

def train_step(generator, discriminator, input_sentence, gen_opt, disc_opt, lambda_div=config.LAMBDA_DIV, gamma=config.GAMMA):
    batch_size = input_sentence.shape[0]
    k = 3  # Number of different paraphrases to generate
    z_samples = tf.random.normal((batch_size * k, config.NOISE_DIM))
    
    with tf.GradientTape(persistent=True) as tape:
        # Generate paraphrases for each sample of z
        generated_texts = generator([input_sentence, z_samples], training=True)
        generated_texts = tf.reshape(generated_texts, (batch_size, k, -1))
        
        # Discriminator's real and fake decisions
        real_output = discriminator(input_sentence, training=True)
        fake_output = discriminator(generated_texts, training=True)
        
        # Calculate the diversity loss
        div_loss = diversity_loss(generated_texts, z_samples, lambda_div)
        
        # Generator and discriminator loss
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
        
        # Combine the losses
        total_gen_loss = gen_loss + gamma * div_loss  # gamma is the weight for the diversity loss
    
    # Calculate the gradients and update the weights for the generator
    gradients_of_generator = tape.gradient(total_gen_loss, generator.trainable_variables)
    gen_opt.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    # Calculate the gradients and update the weights for the discriminator
    gradients_of_discriminator = tape.gradient(disc_loss, discriminator.trainable_variables)
    disc_opt.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    
    # Delete the persistent tape manually to free the resources
    del tape
