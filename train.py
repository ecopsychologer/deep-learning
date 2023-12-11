import tensorflow as tf
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, LeakyReLU, Dropout
from tensorflow.keras import Sequential
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

def log_to_tensorboard(writer, name, text, step):
    with writer.as_default():
        tf.summary.text(name, data=text, step=step)

def train(generator, discriminator, dataset, start_epoch, writer):
    # set up optimizers
    generator_optimizer = tf.keras.optimizers.Adam(config.GEN_LEARN_RATE)
    discriminator_optimizer = tf.keras.optimizers.Adam(config.DISC_LEARN_RATE)
    # Setup the binary cross entropy loss
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    
    def build_feature_extractor():
        model = Sequential()
        for layer in discriminator.layers[:config.FEAT_XTRCTR_LAYERS]:
            model.add(layer)
        return model

    feature_extractor = build_feature_extractor()
    
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
    
    # Average metrics trackers
    avg_gen_loss_tracker = tf.keras.metrics.Mean(name='avg_gen_loss')
    avg_disc_loss_tracker = tf.keras.metrics.Mean(name='avg_disc_loss')
    avg_real_accuracy_tracker = tf.keras.metrics.Mean(name='avg_real_accuracy')
    avg_fake_accuracy_tracker = tf.keras.metrics.Mean(name='avg_fake_accuracy')
    
    with writer.as_default():
        for epoch in range(start_epoch, config.EPOCHS):
            start_time = time.time()
            for image_batch in dataset:
                # Ensure consistent batch size for the last batch
                if image_batch.shape[0] != config.BATCH_SIZE:
                    continue  # Skip the last incomplete batch
                noise = tf.random.normal([config.BATCH_SIZE, config.NOISE_DIM])
                
                with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                    generated_images = generator(noise, training=True)

                    real_output = discriminator(image_batch, training=True)
                    fake_output = discriminator(generated_images, training=True)
                    
                    # Adding noise to labels to soften
                    real_label_noise = tf.random.uniform(shape=tf.shape(real_output), minval=config.REAL_NOISE_MIN_VAL, maxval=config.REAL_NOISE_MAX_VAL)
                    fake_label_noise = tf.random.uniform(shape=tf.shape(fake_output), minval=config.FAKE_NOISE_MIN_VAL, maxval=config.FAKE_NOISE_MAX_VAL)
                    
                    # Get feature representations
                    real_features = feature_extractor(image_batch)
                    fake_features = feature_extractor(generated_images)
                    
                    # Calculate feature matching loss
                    feature_loss = tf.reduce_mean(tf.abs(real_features - fake_features))
                    
                    # Combine with original generator loss
                    gen_loss = generator_loss(fake_output) + config.LAMBDA_FEATURE * feature_loss
                    disc_loss = discriminator_loss(real_output, fake_output, real_label_noise, fake_label_noise)

                gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
                gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

                generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
                discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
                
                # Calculate and log discriminator accuracy
                real_accuracy = tf.reduce_mean(tf.cast(tf.less(real_output, 0.5), tf.float32))
                fake_accuracy = tf.reduce_mean(tf.cast(tf.greater(fake_output, 0.5), tf.float32))
                
                # Track average losses and accuracies
                avg_gen_loss_tracker.update_state(gen_loss)
                avg_disc_loss_tracker.update_state(disc_loss)
                avg_real_accuracy_tracker.update_state(real_accuracy)
                avg_fake_accuracy_tracker.update_state(fake_accuracy)

            if (epoch % config.CHECKPOINT_INTERVAL) == 0 and epoch != start_epoch:
                saveNload.save_model(generator, discriminator, epoch, writer)

            # Log the time it takes for each epoch
            duration = time.time() - start_time
            print(f'Epoch {epoch+1}/{config.EPOCHS} completed in {duration:.2f} seconds')
            name = "Epoch Status"
            text = "Epoch " + str(epoch+1) +"/" + str(config.EPOCHS) + " completed in " + str(duration) + " seconds"
            log_to_tensorboard(writer, name, text, epoch+1)
            
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