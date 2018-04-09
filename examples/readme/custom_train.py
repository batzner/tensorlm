import tensorflow as tf
from tensorlm import Vocabulary, Dataset, GeneratingLSTM

BATCH_SIZE = 20
NUM_TIMESTEPS = 15

with tf.Session() as session:
    # Generate a token -> id vocabulary based on the text
    vocab = Vocabulary.create_from_text("datasets/sherlock/tinytrain.txt", max_vocab_size=96,
                                        level="char")

    # Obtain input and target batches from the text file
    dataset = Dataset("datasets/sherlock/tinytrain.txt", vocab, BATCH_SIZE, NUM_TIMESTEPS)

    # Create the model in a TensorFlow graph
    model = GeneratingLSTM(vocab_size=vocab.get_size(), neurons_per_layer=100, num_layers=2,
                           max_batch_size=BATCH_SIZE, output_keep_prob=0.5)

    # Initialize all defined TF Variables
    session.run(tf.global_variables_initializer())

    # Do the training
    epoch = 1
    step = 1
    for epoch in range(20):
        for inputs, targets in dataset:
            loss = model.train_step(session, inputs, targets)

            if step % 100 == 0:
                # Evaluate from time to time
                dev_dataset = Dataset("datasets/sherlock/tinyvalid.txt", vocab,
                                      batch_size=BATCH_SIZE, num_timesteps=NUM_TIMESTEPS)
                dev_loss = model.evaluate(session, dev_dataset)
                print("Epoch: %d, Step: %d, Train Loss: %f, Dev Loss: %f" % (
                    epoch, step, loss, dev_loss))

                # Sample from the model from time to time
                print("Sampled: \"The " + model.sample_text(session, vocab, "The ") + "\"")

            step += 1
