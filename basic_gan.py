import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Reshape, GaussianNoise, BatchNormalization, Dropout, Layer
from tensorflow.keras import Sequential
import matplotlib.pyplot as plt
from tensorboard.program import TensorBoard
import argparse, time, os, glob, re, stat

""" variables to adjust """
# These are the number of units in the dense layers of your generator and discriminator models. Increasing these can give the network more capacity to learn complex patterns, but too much complexity can lead to overfitting or longer training times.
gen_complexity = 500
disc_complexity = 120

# Define the paths for saving weights
gen_weights_path = "./gen"
disc_weights_path = "./disc"
# Log Path
log_dir = "logs/"

""" values below do NOT require a reset """
# These control how quickly the generator and discriminator learn. Too high, and they may overshoot optimal solutions; too low, and they may get stuck or learn very slowly.
# If the discriminator learns too fast, it may overfit to the current generator's output and not provide useful gradients. If the generator's learning rate is too low in comparison, it may not catch up, leading to poor image quality.
gen_learn_rate = 0.002
disc_learn_rate = 0.00005 # lower rate for the discriminator helps generator
# Larger batches provide more stable gradients but may require more memory and computational power, while smaller batches can sometimes encourage diversity in the generated images and can lead to faster convergence but may also introduce more noise into the training process.
BATCH_SIZE = 170
# The noise added to the labels helps to prevent the discriminator from becoming too confident. However, too much noise can destabilize training.
fake_noise_val = 0.05
real_noise_val = 0.15
# lowering disc_confidence can help the generator learn better
disc_confidence = 0.8

#------------------------------------------------------------------------------#
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

def create_console_space():
    print(f"")
    print(f"\\**********||=+=||**********//")

def clear_logs_and_checkpoints():
    create_console_space()
    
    # Clear TensorBoard logs in the directory
    if os.path.exists(log_dir):
        for file in glob.glob(f'./{log_dir}*'):
            os.remove(file)
        print(f"Cleared TensorBoard logs in {log_dir}")
    else:
        print(f"Attempted to clear logs, but none exist in: {log_dir}")
        # create log directory
        os.makedirs(log_dir, exist_ok=True)
        os.chmod(log_dir, stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
        print(f"Created empty log directory: {log_dir}")

    # Clear generator and discriminator checkpoint files
    for file in glob.glob('./gen_epoch_*.index') + glob.glob('./disc_epoch_*.index'):
        os.remove(file)
        data_file = file.replace('.index', '.data-00000-of-00001')
        if os.path.exists(data_file):
            os.remove(data_file)
        print(f"Removed checkpoint file: {file} and {data_file}")
        
    # Clear Checkpoint file
    if os.path.exists("./checkpoint"):
        os.remove("./checkpoint")
        print(f"Removed checkpoint file.")
        
#------------------------------------------------------------------------------#
num_examples_to_generate = 16  # Number of images to generate for visualization
noise_dim = 100  # Dimensionality of the noise vector

def build_generator():
    model = Sequential([
        GaussianNoise(0.115, input_shape=(noise_dim,)),  # Add noise to input
        Dense(gen_complexity, activation='relu', input_shape=(100,)),  # 100-dimensional noise
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
        Dense(disc_complexity, activation='LeakyReLU'),
        Dropout(0.4),  # Add dropout
        MiniBatchDiscrimination(num_kernels=50, kernel_dim=5),
        Dense(1, activation='sigmoid')
    ])
    return model
#------------------------------------------------------------------------------#
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
#------------------------------------------------------------------------------#
def start_tensorboard(logdir, port=6006):
    tb = TensorBoard()
    tb.configure(argv=[None, '--logdir', logdir, '--port', str(port)])
    url = tb.launch()
    create_console_space()
    print(f"TensorBoard started at {url}")

#------------------------------------------------------------------------------#
def find_latest_epoch():
    gen_files = glob.glob('./gen_epoch_*.index')
    epochs = [int(re.search(r'gen_epoch_(\d+).index', file).group(1)) for file in gen_files]
    return max(epochs) if epochs else None

def save_model_weights(gen, disc, epoch):
    create_console_space()
    gen_weights_path = f"./gen_epoch_{epoch}"
    disc_weights_path = f"./disc_epoch_{epoch}"
    gen.save_weights(gen_weights_path)
    disc.save_weights(disc_weights_path)
    print(f"Checkpoint saved for epoch {epoch}")

def load_model_weights(gen, disc):
    create_console_space()
    latest_epoch = find_latest_epoch()
    if latest_epoch is not None:
        gen_weights_path = f"./gen_epoch_{latest_epoch}"
        disc_weights_path = f"./disc_epoch_{latest_epoch}"
        print(f"Restoring generator from {gen_weights_path}")
        gen.load_weights(gen_weights_path)
        print(f"Restoring discriminator from {disc_weights_path}")
        disc.load_weights(disc_weights_path)
        print(f"Model weights restored from epoch {latest_epoch}")
    else:
        print("No saved model weights found, starting from scratch.")


#----------------------------- train.py -----------------------------#
def train(generator, gen_opt, discriminator, disc_opt, dataset, start_epoch, epochs, writer):
    with writer.as_default():
        for epoch in range(start_epoch, epochs):
            start_time = time.time()
            for image_batch in dataset:
                noise = tf.random.normal([BATCH_SIZE, 100])

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
                save_model_weights(generator, discriminator, epoch)

            # Log the time it takes for each epoch
            duration = time.time() - start_time
            print(f'Epoch {epoch+1}/{epochs} completed in {duration:.2f} seconds')

            # Log the losses to TensorBoard
            with writer.as_default():
                tf.summary.scalar('Generator Loss', gen_loss, step=epoch)
                tf.summary.scalar('Discriminator Loss', disc_loss, step=epoch)
                writer.flush()
            if (epoch % 5) == 0:
                generate_and_save_images(generator, epoch, seed, writer)

    # Generate after the final epoch
    generate_and_save_images(generator, epochs, seed, writer)
    exit

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)

def discriminator_loss(real_labels, fake_labels, real_output, fake_output):
    real_loss = cross_entropy(real_labels*disc_confidence, real_output)
    fake_loss = cross_entropy(fake_labels, fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def discriminator_loss(real_output, fake_output, real_label_noise, fake_label_noise):
    noisy_real_labels = tf.ones_like(real_output) - real_label_noise
    noisy_fake_labels = tf.zeros_like(fake_output) + fake_label_noise

    real_loss = cross_entropy(noisy_real_labels * disc_confidence, real_output)
    fake_loss = cross_entropy(noisy_fake_labels, fake_output)
    
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)
#------------------------------------------------------------------------------#

BUFFER_SIZE = 60000

train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
seed = tf.random.normal([num_examples_to_generate, noise_dim])

EPOCHS = 5000 # enough to run all night, I hope

def main(reset=False):
    latest_epoch = find_latest_epoch()
    # If reset is True, remove the checkpoints, else load them
    if reset:
        clear_logs_and_checkpoints()
        latest_epoch = None
    
    # Initialize models
    generator = build_generator()
    discriminator = build_discriminator()
    # Load if a checkpoint is available
    if latest_epoch is not None:
        load_model_weights(generator, discriminator)
    generator_optimizer = tf.keras.optimizers.Adam(gen_learn_rate)
    discriminator_optimizer = tf.keras.optimizers.Adam(disc_learn_rate)

    # Start tensorboard
    start_tensorboard(log_dir)
    summary_writer = tf.summary.create_file_writer(log_dir)
    # Start training
    start_epoch = latest_epoch if latest_epoch is not None else 0
    train(generator, generator_optimizer, discriminator, discriminator_optimizer, train_dataset, start_epoch, EPOCHS, summary_writer)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reset', action='store_true', help='Clear checkpoint directory and start training from scratch')
    args = parser.parse_args()

    main(reset=args.reset)