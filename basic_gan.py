import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Reshape
from tensorflow.keras import Sequential
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.program import TensorBoard
import os
import glob

def clear_old_images(folder_path='.', extension='png'):
    # Find all .png files in the specified folder
    files = glob.glob(os.path.join(folder_path, f'*.{extension}'))
    for f in files:
        try:
            os.remove(f)
            print(f"Deleted {f}")
        except OSError as e:
            print(f"Error: {e.strerror}")

num_examples_to_generate = 16  # Number of images to generate for visualization
noise_dim = 100  # Dimensionality of the noise vector

# increased complexity helps it learn
gen_complexity = 256
build_complexity = 128

def build_generator():
    model = Sequential([
        Dense(gen_complexity, activation='relu', input_shape=(100,)),  # 100-dimensional noise
        Dense(784, activation='sigmoid'),  # Reshape to 28x28 image
        Reshape((28, 28))
    ])
    return model
def build_discriminator():
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(build_complexity, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    return model

def load_data():
    (train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()
    # Normalize the images to [-1, 1]
    train_images = train_images.reshape(train_images.shape[0], 28, 28).astype('float32')
    train_images = (train_images - 127.5) / 127.5
    return train_images

train_images = load_data()

def generate_and_save_images(model, epoch, test_input, writer):
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    # Save the images for TensorBoard
    with writer.as_default():
        tf.summary.image("Generated Images", plot_to_image(fig), step=epoch)

    plt.close(fig)  # Close the figure to free memory

def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    import io
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


log_dir = "logs/"
summary_writer = tf.summary.create_file_writer(log_dir)

def start_tensorboard(logdir, port=6006):
    tb = TensorBoard()
    tb.configure(argv=[None, '--logdir', logdir, '--port', str(port)])
    url = tb.launch()
    print(f"TensorBoard started at {url}")

def train(generator, discriminator, dataset, epochs, writer):
    with writer.as_default():
        for epoch in range(epochs):
            for image_batch in dataset:
                noise = tf.random.normal([BATCH_SIZE, 100])

                with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                    generated_images = generator(noise, training=True)

                    real_output = discriminator(image_batch, training=True)
                    fake_output = discriminator(generated_images, training=True)
                    
                    # Adding noise to labels
                    real_label_noise = tf.random.uniform(shape=tf.shape(real_output), minval=0.0, maxval=0.3)
                    fake_label_noise = tf.random.uniform(shape=tf.shape(fake_output), minval=0.0, maxval=0.3)
                    noisy_real_labels = tf.ones_like(real_output) - real_label_noise
                    noisy_fake_labels = tf.zeros_like(fake_output) + fake_label_noise

                    gen_loss = generator_loss(fake_output)
                    disc_loss = discriminator_loss(noisy_real_labels, noisy_fake_labels, real_label_noise, fake_label_noise)

                gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
                gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

                generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
                discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
                
                with tf.summary.record_if(epoch % 5 == 0):
                    tf.summary.scalar('gen_loss', gen_loss, step=epoch)
                    tf.summary.scalar('disc_loss', disc_loss, step=epoch)
            # Save the model every few epochs
            if (epoch + 1) % 25 == 0 or epoch == EPOCHS - 1:
                generate_and_save_images(generator, epoch + 1, seed, writer)

    # Generate after the final epoch
    generate_and_save_images(generator, EPOCHS, seed)

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)

# lowering disc_confidence can help the generator learn better
disc_confidence = 0.9
def discriminator_loss(real_labels, fake_labels, real_output, fake_output):
    real_loss = cross_entropy(real_labels*disc_confidence, real_output)
    fake_loss = cross_entropy(fake_labels, fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

gen_learn_rate = 1e-4
generator_optimizer = tf.keras.optimizers.Adam(gen_learn_rate)
disc_learn_rate = 1e-5 # lower rate for the discriminator helps generator
discriminator_optimizer = tf.keras.optimizers.Adam(disc_learn_rate)

generator = build_generator()
discriminator = build_discriminator()

BATCH_SIZE = 256
BUFFER_SIZE = 60000

train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
seed = tf.random.normal([num_examples_to_generate, noise_dim])

EPOCHS = 1000

summary_writer = tf.summary.create_file_writer(log_dir)

# clean pics
clear_old_images()
# start tensorboard
start_tensorboard(log_dir)
# start training
train(generator, discriminator, train_dataset, EPOCHS, summary_writer)

