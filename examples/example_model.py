import tensorflow as tf

from tensorlm import Vocabulary, DatasetIterator, GeneratingLSTM

TEXT_PATH = "datasets/sherlock/train.txt"
DEV_PATH = "datasets/sherlock/valid.txt"
BATCH_SIZE = 20
NUM_TIMESTEPS = 30

with tf.Session() as session:
    vocab = Vocabulary.create_from_text(TEXT_PATH, max_vocab_size=96, level="char")
    dataset = DatasetIterator(TEXT_PATH, vocab, BATCH_SIZE, NUM_TIMESTEPS)

    model = GeneratingLSTM(vocab_size=vocab.get_size(),
                           neurons_per_layer=100,
                           num_layers=2,
                           max_batch_size=BATCH_SIZE)

    session.run(tf.global_variables_initializer())

    epoch = 1
    step = 1
    while epoch < 20:
        for inputs, targets in dataset:
            loss = model.train_step(session, inputs, targets)

            if step % 50 == 0:
                print("Evaluating")
                dev_dataset = DatasetIterator(DEV_PATH, vocab, batch_size=1,
                                              num_timesteps=NUM_TIMESTEPS)
                dev_loss = model.evaluate(session, dev_dataset)
                print(epoch, step, loss, dev_loss)
                print("The " + model.sample(session, vocab, "The "))

            step += 1
