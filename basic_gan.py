import tensorflow as tf
import logging
from tensorboard.program import TensorBoard
import argparse, train, config, saveNload
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
tf.get_logger().setLevel(logging.ERROR)

def start_tensorboard(logdir, port=6006):
    tb = TensorBoard()
    tb.configure(argv=[None, '--logdir', logdir, '--port', str(port), '--bind_all'])
    url = tb.launch()
    config.create_console_space()
    print(f"TensorBoard started at {url}\n")

def main(reset=False):
    latest_epoch = saveNload.find_latest_epoch()
    # If reset is True, remove the checkpoints, else load them
    if reset:
        saveNload.clear_logs_and_checkpoints()
        latest_epoch = None
    
    # Initialize models
    generator = train.build_generator()
    discriminator = train.build_discriminator()
    gan = train.define_gan(generator, discriminator)
    
    # Load if a checkpoint is available
    if latest_epoch is not None:
        generator, discriminator = saveNload.load_model(latest_epoch)

    # Start tensorboard
    start_tensorboard(config.LOG_DIR)
    summary_writer = tf.summary.create_file_writer(config.LOG_DIR)
    # Start training
    start_epoch = latest_epoch if latest_epoch is not None else 0
    train.train(generator, discriminator, gan, saveNload.train_images, start_epoch, summary_writer)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reset', action='store_true', help='Clear checkpoint directory and start training from scratch')
    args = parser.parse_args()

    main(reset=args.reset)