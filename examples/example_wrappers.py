import tensorflow as tf

from tensorlm import CharLM

with tf.Session() as session:
    model = CharLM(session, "datasets/sherlock/train.txt", max_vocab_size=96, neurons_per_layer=100,
                   num_layers=3, num_timesteps=15, save_dir="out")
    model.train(session, max_epochs=5, max_steps=5000, print_every=100, save_interval_hours=0.01)

    generated = model.complete(session, "The ")
    print("The " + generated)

    dev_loss = model.evaluate(session, "datasets/sherlock/valid.txt")
    print("Validation loss: {}".format(dev_loss))
