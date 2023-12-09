import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Reshape, GaussianNoise, BatchNormalization, Dropout, Layer
from tensorflow.keras import Sequential
from tensorboard.program import TensorBoard
import argparse, os, glob, re, stat, train, config, saveNload

""" global variables that require reset """
gen_complexity = config.GEN_COMPLEXITY
disc_complexity = config.DISC_COMPLEXITY
gen_weights_path = config.GEN_WEIGHTS_PATH
disc_weights_path = config.DISC_WEIGHTS_PATH
log_dir = config.LOG_DIR

""" global values that do not require a reset """
gen_learn_rate = config.GEN_LEARN_RATE
disc_learn_rate = config.DISC_LEARN_RATE 
fake_noise_val = config.FAKE_NOISE_VAL
real_noise_val = config.REAL_NOISE_VAL
disc_confidence = config.DISC_CONFIDENCE

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

def clear_logs_and_checkpoints():
    config.create_console_space()
    
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
def start_tensorboard(logdir, port=6006):
    tb = TensorBoard()
    tb.configure(argv=[None, '--logdir', logdir, '--port', str(port)])
    url = tb.launch()
    config.create_console_space()
    print(f"TensorBoard started at {url}")

#------------------------------------------------------------------------------#
def find_latest_epoch():
    gen_files = glob.glob('./gen_epoch_*.index')
    epochs = [int(re.search(r'gen_epoch_(\d+).index', file).group(1)) for file in gen_files]
    return max(epochs) if epochs else None

def save_model_weights(gen, disc, epoch):
    config.create_console_space()
    gen_weights_path = f"./gen_epoch_{epoch}"
    disc_weights_path = f"./disc_epoch_{epoch}"
    gen.save_weights(gen_weights_path)
    disc.save_weights(disc_weights_path)
    print(f"Checkpoint saved for epoch {epoch}")

def load_model_weights(gen, disc):
    config.create_console_space()
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

#------------------------------------------------------------------------------#

train_dataset = tf.data.Dataset.from_tensor_slices(saveNload.train_images).shuffle(config.BUFFER_SIZE).batch(config.BATCH_SIZE)
seed = tf.random.normal([num_examples_to_generate, noise_dim])

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
    train.train(generator, generator_optimizer, discriminator, discriminator_optimizer, train_dataset, start_epoch, config.EPOCHS, summary_writer)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reset', action='store_true', help='Clear checkpoint directory and start training from scratch')
    args = parser.parse_args()

    main(reset=args.reset)