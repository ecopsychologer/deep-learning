import tensorflow as tf
import matplotlib.pyplot as plt
from numpy import expand_dims
from keras.datasets.mnist import load_data
import config, glob, re, config, os, stat, train
from numpy.random import randn

# load and prepare mnist training images
def load_real_samples():
    # load mnist dataset
    (trainX, _), (_, _) = load_data()
    # expand to 3d, e.g. add channels dimension
    X = expand_dims(trainX, axis=-1)
    # convert from unsigned ints to floats
    X = X.astype('float32')
    # scale from [0,255] to [0,1]
    X = X / 255.0
    return X

train_images = load_real_samples()

def log_and_save_plot(examples, epoch, writer, n=5):
    fig = plt.figure(figsize=(n, n))
    # plot images
    for i in range(n * n):
        # define subplot
        plt.subplot(n, n, 1 + i)
        # turn off axis
        plt.axis('off')
        # plot raw pixel data
        plt.imshow(examples[i, :, :, 0], cmap='gray_r')
        # save plot to file
    filename = 'generated_plot_e%03d.png' % (epoch+1)
    plt.savefig(filename)
    # Save the images for TensorBoard
    with writer.as_default():
        tf.summary.image("Generated Images", plot_to_image(fig), step=epoch)
        writer.flush()
    plt.close()

def generate_and_save_images(epoch, g_model, latent_dim, writer, n_samples=100):
    # generate points in latent space
    x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(n_samples, latent_dim)
    # predict outputs
    X = g_model.predict(x_input)
    log_and_save_plot(X, epoch, writer)

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

def find_latest_epoch():
    gen_path = config.GEN_MODEL_PATH + "*" + config.CHECKPOINT_EXT
    gen_files = glob.glob(gen_path)
    epochs = [int(re.search(fr'{config.GEN_MODEL_PATH}(\d+){config.CHECKPOINT_EXT}', file).group(1)) for file in gen_files]
    return max(epochs) if epochs else None

def save_model(gen, disc, epoch, writer):
    config.create_console_space()
    gen_model_path = f"{config.GEN_MODEL_PATH}{epoch}{config.CHECKPOINT_EXT}"
    disc_model_path = f"{config.DISC_MODEL_PATH}{epoch}{config.CHECKPOINT_EXT}"
    
    # Save the entire model to a file
    gen.save(gen_model_path)
    disc.save(disc_model_path)
    
    print(f"Full model saved for epoch {epoch}")
    name = "Model Save Status"
    text = "Checkpoint  " + str(epoch/config.CHECKPOINT_INTERVAL) +"/" + str(config.EPOCHS/config.CHECKPOINT_INTERVAL) + " completed."
    train.log_to_tensorboard(writer, name, text, epoch)

def load_model(epoch):
    config.create_console_space()
    gen_model_path = f"{config.GEN_MODEL_PATH}{epoch}{config.CHECKPOINT_EXT}"
    disc_model_path = f"{config.DISC_MODEL_PATH}{epoch}{config.CHECKPOINT_EXT}"
    
    # Load the entire model from the file
    gen_loaded = tf.keras.models.load_model(gen_model_path)
    disc_loaded = tf.keras.models.load_model(disc_model_path)
    
    print(f"Models restored from epoch {epoch}")
    
    return gen_loaded, disc_loaded
        
def clear_logs_and_checkpoints():
    config.create_console_space()
    log_dir = config.LOG_DIR
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

    gen_model_path = config.GEN_MODEL_PATH + "*" + config.CHECKPOINT_EXT
    disc_model_path = config.DISC_MODEL_PATH + "*" + config.CHECKPOINT_EXT
    # Clear checkpoint files
    for file in glob.glob(gen_model_path) + glob.glob(disc_model_path):
        os.remove(file)
        print(f"Removed checkpoint file: {file}")
        
    # Clear Checkpoint file
    if os.path.exists("./checkpoint"):
        os.remove("./checkpoint")
        print(f"Removed the \'checkpoint\' file.")
    print("\n")