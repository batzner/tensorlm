import numpy as np
import tensorflow as tf

from tensorlm.dataset import Vocabulary, DatasetIterator
from tensorlm.model import GeneratingLSTM


class _BaseLM:
    def __init__(self, level, tf_session, train_text_path, max_vocab_size, neurons_per_layer,
                 num_layers, max_batch_size, num_timesteps):
        self.train_text_path = train_text_path
        self.max_batch_size = max_batch_size
        self.num_timesteps = num_timesteps

        # Build the vocabulary to determine the actual vocabulary size
        self.vocab = Vocabulary.create_from_text(train_text_path, max_vocab_size=max_vocab_size,
                                                 level=level)

        self.model = GeneratingLSTM(vocab_size=self.vocab.get_size(),
                                    neurons_per_layer=neurons_per_layer, num_layers=num_layers,
                                    max_batch_size=max_batch_size)

        tf_session.run(tf.global_variables_initializer())

    def train(self, tf_session, max_epochs=10, max_steps=None, batch_size=None, text_path=None,
              print_every=None):
        if not text_path:
            text_path = self.train_text_path

        # Clip the batch size
        if not batch_size or batch_size > self.max_batch_size:
            batch_size = self.max_batch_size

        epoch = 1
        step = 1
        last_losses = []

        while epoch <= max_epochs and (step <= max_steps or not max_steps):
            for inputs, targets in DatasetIterator(text_path, self.vocab, batch_size,
                                                   self.num_timesteps):
                loss = self.model.train_step(tf_session, inputs, targets)
                last_losses.append(loss)

                if print_every and step % print_every == 0:
                    avg_loss = np.mean(last_losses)
                    print("Epoch: {}, Step: {}, Avg. Train Loss: {}".format(epoch, step, avg_loss))
                    last_losses = []

                step += 1

                if step > max_steps or not max_steps:
                    break
            epoch += 1

    def evaluate(self, tf_session, text_path):
        dataset_iter = DatasetIterator(text_path, self.vocab, batch_size=1,
                                       num_timesteps=self.num_timesteps)
        loss = self.model.evaluate(tf_session, dataset_iter)
        return loss

    def decode(self, tf_session, prime):
        return self.model.sample(tf_session, self.vocab, prime)


class CharLM(_BaseLM):
    def __init__(self, tf_session, train_text_path, max_vocab_size=96, neurons_per_layer=100,
                 num_layers=3, max_batch_size=10, num_timesteps=15):
        super().__init__(level="char", tf_session=tf_session, train_text_path=train_text_path,
                         max_vocab_size=max_vocab_size, neurons_per_layer=neurons_per_layer,
                         num_layers=num_layers, max_batch_size=max_batch_size,
                         num_timesteps=num_timesteps)


class WordLM(_BaseLM):
    def __init__(self, tf_session, train_text_path, max_vocab_size=2000, neurons_per_layer=100,
                 num_layers=2, max_batch_size=10, num_timesteps=5):
        super().__init__(level="word", tf_session=tf_session, train_text_path=train_text_path,
                         max_vocab_size=max_vocab_size, neurons_per_layer=neurons_per_layer,
                         num_layers=num_layers, max_batch_size=max_batch_size,
                         num_timesteps=num_timesteps)
