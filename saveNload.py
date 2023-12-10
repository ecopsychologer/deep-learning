import tensorflow as tf
import matplotlib.pyplot as plt
import glob, re, config, os, stat, train

def load_data():
    (train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()
    # Normalize the images to [-1, 1]
    train_images = train_images.reshape(train_images.shape[0], 28, 28).astype('float32')
    train_images = (train_images - 127.5) / 127.5
    return train_images

train_images = load_data()

def generate_and_save_images(model, epoch, test_input, writer):
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(5, 5))

    for i in range(predictions.shape[0]):
        plt.subplot(5, 5, i+1)
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

def find_latest_epoch():
    gen_files = glob.glob('./gen_epoch_*.index')
    epochs = [int(re.search(r'gen_epoch_(\d+).index', file).group(1)) for file in gen_files]
    return max(epochs) if epochs else None

def save_model_weights(gen, disc, epoch, writer):
    config.create_console_space()
    gen_weights_path = f"./gen_epoch_{epoch}"
    disc_weights_path = f"./disc_epoch_{epoch}"
    gen.save_weights(gen_weights_path)
    disc.save_weights(disc_weights_path)
    print(f"Checkpoint saved for epoch {epoch}")
    print("\n")
    name = "Checkpoint"
    text = "Checkpoint  " + str(epoch/config.CHECKPOINT_INTERVAL) +"/" + str(config.EPOCHS/config.CHECKPOINT_INTERVAL) + " completed."
    train.log_to_tensorboard(writer, name, text, epoch)

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
    print("\n")
        
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
        print(f"Removed the \'checkpoint\' file.")
    print("\n")