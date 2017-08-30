"""Example of creating a model - basic usage of the tensorlm package."""

import tensorflow as tf

from tensorlm import CharLM

with tf.Session() as session:
    # Create a new language model
    model = CharLM(session, "datasets/sherlock/tinytrain.txt", max_vocab_size=96,
                   neurons_per_layer=100, num_layers=3, num_timesteps=15, save_dir="out/model")

    # Train it, but evaluate and sample during training so we can see the progress
    model.train(session, max_epochs=20, max_steps=5000, log_interval=100, print_logs=True,
                evaluate_interval=1000, evaluate_text_path="datasets/sherlock/tinyvalid.txt",
                sample_interval=1000, save_interval_hours=0.5)

    # Try out the new model
    generated = model.sample(session, "The ", num_steps=100)
    print("The " + generated)

    dev_loss = model.evaluate(session, "datasets/sherlock/tinyvalid.txt")
    print("Validation loss: {}".format(dev_loss))
