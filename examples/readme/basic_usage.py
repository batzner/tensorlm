import tensorflow as tf
from tensorlm import CharLM

with tf.Session() as session:
    # Create a new model. You can also use WordLM
    model = CharLM(session, "datasets/sherlock/tinytrain.txt", max_vocab_size=96,
                   neurons_per_layer=100, num_layers=3, num_timesteps=15)

    # Train it
    model.train(session, max_epochs=10, max_steps=500)

    # Let it generate a text
    generated = model.sample(session, "The ", num_steps=100)
    print("The " + generated)