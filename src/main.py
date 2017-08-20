import tensorflow as tf

from src.generating_lstm.dataset import Vocabulary, Dataset
from src.generating_lstm.model import GeneratingLSTM

TEXT_PATH = "datasets/sherlock/sherlock-train.txt"

def train():
    with tf.Session() as session:
        vocab = Vocabulary.create_from_text(TEXT_PATH)
        dataset = Dataset(TEXT_PATH, vocab, 128)

        for inputs, targets in dataset:
            print(inputs)
            print(targets)
            print("--")

        model = GeneratingLSTM(vocab_size=98,
                               num_neurons=100,
                               num_layers=3,
                               max_batch_size=128)

        epoch = 1
        while epoch < 20:
            pass


def main(_):
    train()


if __name__ == "__main__":
    tf.app.flags.DEFINE_string("config_path", "src/char_lm/config/dev.json", "Configuration path")
    FLAGS = tf.app.flags.FLAGS

    tf.app.run()
