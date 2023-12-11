import tensorflow as tf
import numpy as np
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, LeakyReLU, Dropout
from tensorflow.keras import Sequential
from numpy import ones, vstack
import config, saveNload, time

seed = tf.random.normal([config.NUM_EXAMPLES_TO_GEN, config.NOISE_DIM])

# switch to convolutional approach
def build_generator(latent_dim=config.LATENT_DIM):
    model = Sequential()
    # foundation for 7x7 image
    n_nodes = config.GEN_NODES * 7 * 7
    model.add(Dense(n_nodes, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=config.GEN_ALPHA))
    model.add(Reshape((7, 7, config.GEN_NODES)))
    # upsample to 14x14
    model.add(Conv2DTranspose(config.GEN_NODES, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=config.GEN_ALPHA))
    # upsample to 28x28
    model.add(Conv2DTranspose(config.GEN_NODES, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=config.GEN_ALPHA))
    model.add(Conv2D(1, (7,7), activation='sigmoid', padding='same'))
    return model

def build_discriminator(in_shape=(28,28,1)):
    model = Sequential()
    model.add(Conv2D(config.DISC_NODES, (3,3), strides=(2, 2), padding='same', input_shape=in_shape))
    model.add(LeakyReLU(alpha=config.DISC_ALPHA))
    model.add(Dropout(config.DROPOUT_RATE))
    model.add(Conv2D(config.DISC_NODES, (3,3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=config.DISC_ALPHA))
    model.add(Dropout(config.DROPOUT_RATE))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    opt = Adam(learning_rate=config.DISC_LEARN_RATE, beta_1=config.DISC_BETA_1)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

# define the combined generator and discriminator model, for updating the generator
def define_gan(generator, discriminator):
    # lock discriminator
    discriminator.trainable = False
    # connect them
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    # compile model
    opt = Adam(learning_rate=config.GAN_LEARN_RATE, beta_1=config.GAN_BETA_1)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model

def interpolate_latent_points(start_points, end_points, num_steps=config.INTERPOLATION_STEPS, overlap=3):
    interpolated_points = []
    for i in range(num_steps + overlap):
        alpha = i / float(num_steps)
        interpolated = alpha * end_points + (1 - alpha) * start_points
        interpolated_points.append(interpolated)
    return np.array(interpolated_points[overlap-1:-1])  # Skip the first 'overlap-1' points and the last point

# Setup the binary cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)

# Define loss functions
def discriminator_loss(real_output, fake_output, real_label_noise, fake_label_noise):
    noisy_real_labels = tf.ones_like(real_output) - real_label_noise
    noisy_fake_labels = tf.zeros_like(fake_output) + fake_label_noise

    real_loss = cross_entropy(noisy_real_labels * config.DISC_CONFIDENCE, real_output)
    fake_loss = cross_entropy(noisy_fake_labels, fake_output)
    
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def build_feature_extractor(in_model):
    model = Sequential()
    for layer in in_model.layers[:config.FEAT_XTRCTR_LAYERS]:
        model.add(layer)
    return model

def log_to_tensorboard(writer, name, text, step):
    with writer.as_default():
        tf.summary.text(name, data=text, step=step)

def train(generator, discriminator, gan, dataset, start_epoch, writer):
    # set up feature extractor
    feature_extractor = build_feature_extractor(discriminator)
    
    # Average metrics trackers
    avg_gen_loss_tracker = tf.keras.metrics.Mean(name='avg_gen_loss')
    avg_disc_loss_tracker = tf.keras.metrics.Mean(name='avg_disc_loss')
    
    batch_per_epoch = int(dataset.shape[0] / config.BATCH_SIZE)
    half_batch = int(config.BATCH_SIZE / 2)
    
    with writer.as_default():
        for epoch in range(start_epoch, config.EPOCHS):
            start_time = time.time()
            for j in range(batch_per_epoch):
                # Generate noise for a whole batch
                noise = tf.random.normal([config.BATCH_SIZE, config.NOISE_DIM])
                # Generate fake images
                generated_images = generator(noise, training=True)
                
                # Get randomly selected 'real' samples
                X_real, y_real = saveNload.generate_real_samples(dataset, half_batch)
                # Generate 'fake' examples
                X_fake, y_fake = saveNload.generate_fake_samples(generator, config.LATENT_DIM, half_batch)
                # Create training set for the discriminator merging the two above
                X, y = vstack((X_real, X_fake)), vstack((y_real, y_fake))

                # Update discriminator model weights
                if (config.FEATURE_MATCHING): # feature matching is currently broken but I want to have the option to implement it
                    # set up optimizers
                    generator_optimizer = tf.keras.optimizers.Adam(config.GEN_LEARN_RATE)
                    discriminator_optimizer = tf.keras.optimizers.Adam(config.DISC_LEARN_RATE)
                    
                    with tf.GradientTape() as disc_tape:
                        real_output = discriminator(X_real, training=True)
                        fake_output = discriminator(X_fake, training=True)

                        # Get feature representations
                        real_features = feature_extractor(X_real)
                        fake_features = feature_extractor(X_fake)

                        # Calculate feature matching loss
                        feature_loss = tf.reduce_mean(tf.abs(real_features - fake_features))

                        # Calculate total discriminator loss
                        d_loss = discriminator_loss(real_output, fake_output, y_real, y_fake) + config.LAMBDA_FEATURE * feature_loss

                    # Calculate gradients and update discriminator weights
                    gradients_of_discriminator = disc_tape.gradient(d_loss, discriminator.trainable_variables)
                    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

                    # Update generator model weights
                    with tf.GradientTape() as gen_tape:
                        fake_output = discriminator(generated_images, training=False)
                        g_loss = generator_loss(fake_output)

                    # Calculate gradients and update generator weights
                    gradients_of_generator = gen_tape.gradient(g_loss, generator.trainable_variables)
                    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
                    
                    # Track average losses
                    avg_gen_loss_tracker.update_state(g_loss)
                    avg_disc_loss_tracker.update_state(d_loss)
                else:
                    d_loss, _ = discriminator.train_on_batch(X, y)
                    # prepare points in latent space as input for the generator
                    X_gan = saveNload.generate_latent_points(config.SAMPLES, config.BATCH_SIZE)
                    # create inverted labels for the fake samples
                    y_gan = ones((config.BATCH_SIZE, 1))
                    
                    # update the generator via the discriminator's error 
                    g_loss = gan.train_on_batch(X_gan, y_gan)

                    # Track average losses and accuracies
                    avg_gen_loss_tracker.update_state(g_loss)
                    avg_disc_loss_tracker.update_state(d_loss)
                    
                    # store last latent vector
                    last_latent_vector = X_gan

            if (epoch % config.CHECKPOINT_INTERVAL) == 0 and epoch != start_epoch:
                saveNload.save_model(generator, discriminator, epoch, writer)

            # Log the time it takes for each epoch
            duration = time.time() - start_time
            print(f'Epoch {epoch+1}/{config.EPOCHS} completed in {duration:.2f} seconds')
            name = "Epoch Status"
            text = "Epoch " + str(epoch+1) +"/" + str(config.EPOCHS) + " completed in " + str(duration) + " seconds"
            saveNload.eval_discrim(epoch, generator, discriminator, dataset, writer)
            log_to_tensorboard(writer, name, text, epoch+1)
            
            # Log the losses and accuracies to TensorBoard
            # Log the average metrics for the epoch
            with writer.as_default():
                tf.summary.scalar('Average Generator Loss', avg_gen_loss_tracker.result(), step=epoch)
                tf.summary.scalar('Average Discriminator Loss', avg_disc_loss_tracker.result(), step=epoch)
                writer.flush()
            saveNload.generate_and_save_images(epoch, generator, writer)
            # Reset metrics every epoch
            avg_gen_loss_tracker.reset_states()
            avg_disc_loss_tracker.reset_states()
            
            saveNload.save_latent_vectors(last_latent_vector, epoch)

    # Generate after the final epoch
    saveNload.generate_and_save_images(config.EPOCHS, generator, writer)
    exit