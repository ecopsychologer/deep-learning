""" Config File With Global Variables """

# --- General Settings ---
# Random seed for reproducibility (set to None for random initialization)
RANDOM_SEED = 42

# --- Model Architecture ---
# Number of units in the dense layers of generator and discriminator models
# These control how quickly the generator and discriminator learn. Too high, and they may overshoot optimal solutions; too low, and they may get stuck or learn very slowly.
# If the discriminator learns too fast, it may overfit to the current generator's output and not provide useful gradients. If the generator's learning rate is too low in comparison, it may not catch up, leading to poor image quality.
# These are the number of units in the dense layers of your generator and discriminator models. Increasing these can give the network more capacity to learn complex patterns, but too much complexity can lead to overfitting or longer training times.
GEN_COMPLEXITY = 510
DISC_COMPLEXITY = 77 # lower rate for the discriminator helps generator

# --- Learning Rate & Scheduling ---
# Learning rates for generator and discriminator
GEN_LEARN_RATE = 0.0002
DISC_LEARN_RATE = 0.00002

# Learning rate decay factor (None for no decay)
LR_DECAY_FACTOR = None  # e.g., 0.95
# Epoch interval for applying learning rate decay
LR_DECAY_EPOCHS = None  # e.g., 100

# --- Training Settings ---
# Larger batches provide more stable gradients but may require more memory and computational power, while smaller batches can sometimes encourage diversity in the generated images and can lead to faster convergence but may also introduce more noise into the training process.
BATCH_SIZE = 256
# Total number of epochs for training
EPOCHS = 5000

# --- Data Handling ---
# Buffer size for shuffling data
BUFFER_SIZE = 80000

# --- Regularization Techniques ---
# Dropout rate (None for no dropout)
DROPOUT_RATE = 0.25  # e.g., 0.3

# --- Noise and Confidence Parameters ---
# Noise values for training stability
# The noise added to the labels helps to prevent the discriminator from becoming too confident. However, too much noise can destabilize training.
FAKE_NOISE_VAL = 0.1
REAL_NOISE_VAL = 0.05
# Discriminator's confidence threshold
# lowering disc_confidence can help the generator learn better
DISC_CONFIDENCE = 0.6

# --- Validation & Monitoring ---
# Interval for saving model checkpoints
CHECKPOINT_INTERVAL = 10

# --- File Paths ---
# Paths for saving weights, logs, etc.
GEN_WEIGHTS_PATH = "./gen"
DISC_WEIGHTS_PATH = "./disc"
LOG_DIR = "logs/"

# Number of images to generate for visualization
NUM_EXAMPLES_TO_GEN = 25
# Dimensionality of the noise vector
NOISE_DIM = 100

def create_console_space():
    print("\n\\\\**********||=+=||**********//")

def create_console_space_around(this):
    print("\n\\\\**********||=+=||**********//")
    print(f"{this}")
    print("\n")
