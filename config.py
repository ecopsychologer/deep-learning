""" Config File With Global Variables """

# --- General Settings ---
# Random seed for reproducibility (set to None for random initialization)
RANDOM_SEED = 42

# --- Model Architecture ---
# Number of units in the dense layers of generator and discriminator models
# These control how quickly the generator and discriminator learn. Too high, and they may overshoot optimal solutions; too low, and they may get stuck or learn very slowly.
# If the discriminator learns too fast, it may overfit to the current generator's output and not provide useful gradients. If the generator's learning rate is too low in comparison, it may not catch up, leading to poor image quality.
# These are the number of units in the dense layers of your generator and discriminator models. Increasing these can give the network more capacity to learn complex patterns, but too much complexity can lead to overfitting or longer training times.
GEN_COMPLEXITY = 512
DISC_COMPLEXITY = 128 # lower rate for the discriminator helps generator
GEN_ALPHA = 0.2
GEN_NODES = 128
DROPOUT_RATE = 0.4

DISC_ALPHA = 0.2
DISC_NODES = 64
DISC_BETA_1 = 0.5

GAN_BETA_1 = 0.5

SAMPLES = 100

# --- Learning Rate & Scheduling ---
# Learning rates for generator and discriminator
GEN_LEARN_RATE = 0.003
DISC_LEARN_RATE = 0.0002
GAN_LEARN_RATE = 0.0002

# --- Feature Matching ---
# Enable
FEATURE_MATCHING = False
# Number of layers from the discriminator to use in the feature extractor
FEAT_XTRCTR_LAYERS = 5
# Value for how much to give weight to feature matching loss
LAMBDA_FEATURE = 0.2

# --- Training Settings ---
# Larger batches provide more stable gradients but may require more memory and computational power, while smaller batches can sometimes encourage diversity in the generated images and can lead to faster convergence but may also introduce more noise into the training process.
BATCH_SIZE = 256
# Total number of epochs for training
EPOCHS = 5000
LATENT_DIM = 100

# --- Data Handling ---
# Buffer size for shuffling data
BUFFER_SIZE = 60000



# --- Noise and Confidence Parameters ---
# Noise values for training stability
# The noise added to the labels helps to prevent the discriminator from becoming too confident. However, too much noise can destabilize training.
FAKE_NOISE_MAX_VAL = 0.4
FAKE_NOISE_MIN_VAL = 0.0
REAL_NOISE_MAX_VAL = 1.4
REAL_NOISE_MIN_VAL = 0.6
# Discriminator's confidence threshold
# lowering disc_confidence can help the generator learn better
DISC_CONFIDENCE = 0.5

# --- Validation & Monitoring ---
# Interval for saving model checkpoints
CHECKPOINT_INTERVAL = 10

# --- File Paths ---
# Paths for saving weights, logs, etc.
GEN_WEIGHTS_PATH = "./gen"
DISC_WEIGHTS_PATH = "./disc"
LOG_DIR = "./logs/"
CHECKPOINT_EXT = ".keras"

GEN_MODEL_PRE = "gen_model_epoch_"
DISC_MODEL_PRE = "disc_model_epoch_"
GEN_MODEL_PATH = LOG_DIR + GEN_MODEL_PRE
DISC_MODEL_PATH = LOG_DIR + DISC_MODEL_PRE

LATENT_VECTOR_PATH = LOG_DIR

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
