import sys
import os
import json

import tensorflow as tf

# Model parameters
from tensorlm.wrappers import BaseLM

tf.app.flags.DEFINE_string("level", "char", "Level of tokenization. Either 'char' or 'word'.")
tf.app.flags.DEFINE_integer("max_vocab_size", 1000,
                            "Maximum size of the input and output vocabulary.")
tf.app.flags.DEFINE_integer("neurons_per_layer", 100, "Number of neurons per LSTM-layer")
tf.app.flags.DEFINE_integer("num_layers", 3, "Number of LSTM-layers")
tf.app.flags.DEFINE_integer("max_batch_size", 10, "Maximum number of batches that are fed to the "
                                                  "model at once")
tf.app.flags.DEFINE_integer("num_timesteps", 5, "Number of steps that the LSTM is unrolled for")

# General parameters
tf.app.flags.DEFINE_string("save_dir", None, "Directory where the model is / will be saved")
tf.app.flags.DEFINE_string("evaluate_text_path", None, "Path to .txt file")

# Training parameters
tf.app.flags.DEFINE_boolean("train", False,
                            "Set to True for training a new model or resuming training")
tf.app.flags.DEFINE_string("train_text_path", None, "Path to .txt file")
tf.app.flags.DEFINE_integer("max_epochs", 10, "Maximum number of epochs to train")
tf.app.flags.DEFINE_integer("max_steps", None, "Maximum number of training steps to perform. Use "
                                               "None to ignore this limit.")
tf.app.flags.DEFINE_float("save_interval_hours", 0.5,
                          "Number of hours between saves of the model's trained parameters.")
tf.app.flags.DEFINE_integer("log_interval", 100,
                            "Number of train steps between logs. Set to None to disable logs. Logs "
                            "of the train loss will be stored in the trainlog.json file in the "
                            "save_dir.")
tf.app.flags.DEFINE_integer("train_evaluate_interval", 1000,
                            "Number of train steps between validation runs. Set None to disable "
                            "validation during training. If not None, you need to set "
                            "evaluate_text_path as well.")
tf.app.flags.DEFINE_integer("train_sample_interval", 1000,
                            "Number of train steps between sample runs. Set None to disable sample "
                            "during training. If not None, you need to set train_sample_prime as "
                            "well.")
tf.app.flags.DEFINE_string("train_sample_prime", "The ", "Start of the text that the model will"
                                                         "complete.")

# Rest
tf.app.flags.DEFINE_boolean("sample", False, "Set to True for interactive sampling.")
tf.app.flags.DEFINE_integer("sample_steps", 100, "Generated tokens during interactive sampling.")

tf.app.flags.DEFINE_boolean("evaluate", False,
                            "Set to True for evaluating the performance on the evaluate_text_path.")
FLAGS = tf.app.flags.FLAGS


def sample_interactive(tf_session, model, num_steps):
    # Read the input with >
    sys.stdout.write('> ')
    sys.stdout.flush()
    input_sequence = sys.stdin.readline().strip()
    while input_sequence:
        sampled = model.sample(tf_session, input_sequence, num_steps)
        # Print the sampled text to the console
        print(sampled)
        print('> ', end='')
        sys.stdout.flush()
        input_sequence = sys.stdin.readline().strip()


def main(_):
    with tf.Session() as session:
        model = BaseLM(session, level=FLAGS.level,
                       train_text_path=FLAGS.train_text_path,
                       max_vocab_size=FLAGS.max_vocab_size,
                       neurons_per_layer=FLAGS.neurons_per_layer,
                       num_layers=FLAGS.num_layers,
                       max_batch_size=FLAGS.max_batch_size,
                       num_timesteps=FLAGS.num_timesteps,
                       save_dir=FLAGS.save_dir)

        if FLAGS.train:
            # Store the flags
            flags_path = os.path.join(FLAGS.save_dir, "flags.json")
            if not os.path.exists(flags_path):
                with open(flags_path, "w") as f:
                    json.dump(FLAGS.__dict__["__flags"], f)

            model.train(session, max_epochs=FLAGS.max_epochs,
                        max_steps=FLAGS.max_steps,
                        log_interval=FLAGS.log_interval,
                        evaluate_interval=FLAGS.train_evaluate_interval,
                        evaluate_text_path=FLAGS.evaluate_text_path,
                        sample_interval=FLAGS.train_sample_interval,
                        sample_prime=FLAGS.train_sample_prime,
                        save_interval_hours=FLAGS.save_interval_hours,
                        print_logs=True)

        elif FLAGS.evaluate:
            loss = model.evaluate(session, FLAGS.evaluate_text_path)
            print("Validation loss: {}".format(loss))

        elif FLAGS.sample:
            sample_interactive(session, model, FLAGS.sample_steps)


if __name__ == "__main__":
    tf.app.run()
