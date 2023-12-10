import tensorflow as tf
from tensorboard.program import TensorBoard
import argparse, train, config, saveNload

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


def start_tensorboard(logdir, port=6006, host='192.168.1.49'):
    tb = TensorBoard()
    tb.configure(argv=[None, '--logdir', logdir, '--port', str(port), '--host', host])
    url = tb.launch()
    config.create_console_space()
    print(f"TensorBoard started at {url}")

train_dataset = tf.data.Dataset.from_tensor_slices(saveNload.train_images).shuffle(config.BUFFER_SIZE).batch(config.BATCH_SIZE)

def main(reset=False):
    latest_epoch = saveNload.find_latest_epoch()
    # If reset is True, remove the checkpoints, else load them
    if reset:
        saveNload.clear_logs_and_checkpoints()
        latest_epoch = None
    
    # Initialize models
    generator = train.build_generator()
    discriminator = train.build_discriminator()
    # Load if a checkpoint is available
    if latest_epoch is not None:
        saveNload.load_model_weights(generator, discriminator)
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