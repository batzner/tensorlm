import tensorflow as tf

from tensorlm import CharLM
from tensorlm import WordLM

with tf.Session() as session:
    model = WordLM(session, "datasets/sherlock/train.txt", max_vocab_size=20000,
                   neurons_per_layer=100, num_layers=3, num_timesteps=15, save_dir="out/model")

    model.train(session, max_epochs=5, max_steps=5000, log_interval=100, print_logs=True,
                evaluate_interval=1000, evaluate_text_path="datasets/sherlock/valid.txt",
                sample_interval=1000, save_interval_hours=0.5)

    generated = model.sample(session, "The ", num_steps=100)
    print("The " + generated)

    dev_loss = model.evaluate(session, "datasets/sherlock/valid.txt")
    print("Validation loss: {}".format(dev_loss))
