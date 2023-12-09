""" Config File With Global Variables """
# These control how quickly the generator and discriminator learn. Too high, and they may overshoot optimal solutions; too low, and they may get stuck or learn very slowly.
# If the discriminator learns too fast, it may overfit to the current generator's output and not provide useful gradients. If the generator's learning rate is too low in comparison, it may not catch up, leading to poor image quality.
GEN_LEARN_RATE = 0.0015
DISC_LEARN_RATE = 0.00005 # lower rate for the discriminator helps generator

# Larger batches provide more stable gradients but may require more memory and computational power, while smaller batches can sometimes encourage diversity in the generated images and can lead to faster convergence but may also introduce more noise into the training process.
BATCH_SIZE = 170
# The noise added to the labels helps to prevent the discriminator from becoming too confident. However, too much noise can destabilize training.
FAKE_NOISE_VAL = 0.05
REAL_NOISE_VAL = 0.15
# lowering disc_confidence can help the generator learn better
DISC_CONFIDENCE = 0.8


""" Requires a Reset """
# These are the number of units in the dense layers of your generator and discriminator models. Increasing these can give the network more capacity to learn complex patterns, but too much complexity can lead to overfitting or longer training times.
GEN_COMPLEXITY = 500
DISC_COMPLEXITY = 120
# Define the base paths for saving weights
GEN_WEIGHTS_PATH = "./gen"
DISC_WEIGHTS_PATH = "./disc"
# Log Path
LOG_DIR = "logs/"
EPOCHS = 5000
BUFFER_SIZE = 60000

NUM_EXAMPLES_TO_GEN = 16  # Number of images to generate for visualization
NOISE_DIM = 100  # Dimensionality of the noise vector

def create_console_space():
    print(f"")
    print(f"\\**********||=+=||**********//")



