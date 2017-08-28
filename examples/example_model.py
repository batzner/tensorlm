"""Example of creating a model. For an easier way, see example_wrappers.py"""

import tensorflow as tf

from tensorlm import Vocabulary, Dataset, GeneratingLSTM

TEXT_PATH = "datasets/sherlock/train.txt"
DEV_PATH = "datasets/sherlock/valid.txt"
BATCH_SIZE = 20
NUM_TIMESTEPS = 30

with tf.Session() as session:
    # Generate a vocabulary based on the text
    vocab = Vocabulary.create_from_text(TEXT_PATH, max_vocab_size=96, level="char")

    # Obtain input and target batches from the text file
    dataset = Dataset(TEXT_PATH, vocab, BATCH_SIZE, NUM_TIMESTEPS)

    # Create the model in a TensorFlow graph
    model = GeneratingLSTM(vocab_size=vocab.get_size(),
                           neurons_per_layer=100,
                           num_layers=2,
                           max_batch_size=BATCH_SIZE,
                           output_keep_prob=1)

    # Initialize all defined TF Variables
    session.run(tf.global_variables_initializer())

    # Do the training
    epoch = 1
    step = 1
    for epoch in range(20):
        for inputs, targets in dataset:
            loss = model.train_step(session, inputs, targets)

            if step % 50 == 0:
                # Evaluate from time to time
                print("Evaluating")
                dev_dataset = Dataset(DEV_PATH, vocab, batch_size=1,
                                      num_timesteps=NUM_TIMESTEPS)
                dev_loss = model.evaluate(session, dev_dataset)
                print("Epoch: %d, Step: %d, Train Loss: %f, Dev Loss: %f" % (
                    epoch, step, loss, dev_loss))

                # Sample from the model from time to time
                print("The " + model.sample(session, vocab, "The "))

            step += 1
