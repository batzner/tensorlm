import numpy as np
import tensorflow as tf

from src.generating_lstm.dataset import Vocabulary, Dataset
from src.generating_lstm.model import GeneratingLSTM

TEXT_PATH = "datasets/sherlock/sherlock-train.txt"
BATCH_SIZE = 20
NUM_TIMESTEPS = 10


def train():
    with tf.Session() as session:

        vocab = Vocabulary.create_from_text(TEXT_PATH, max_vocab_size=98)
        model = GeneratingLSTM(vocab_size=vocab.get_size(),
                               num_neurons=24,
                               num_layers=2,
                               max_batch_size=BATCH_SIZE)

        session.run(tf.global_variables_initializer())

        dataset = Dataset(TEXT_PATH, vocab, BATCH_SIZE, NUM_TIMESTEPS)

        epoch = 1
        step = 1
        while epoch < 20:
            for inputs, targets in dataset:
                loss = model.train_step(session, inputs, targets)

                if step % 500 == 0:
                    print(epoch, step, loss)
                    print("The " + model.sample(session, vocab, "The "))
                    print(" INPUTS: " + "".join(vocab.ids_to_tokens(inputs[0])))
                    print("TARGETS: " + "".join(vocab.ids_to_tokens(targets[0])))

                step += 1


def main(_):
    train()


if __name__ == "__main__":
    tf.app.flags.DEFINE_string("config_path", "src/char_lm/config/dev.json", "Configuration path")
    FLAGS = tf.app.flags.FLAGS

    tf.app.run()
