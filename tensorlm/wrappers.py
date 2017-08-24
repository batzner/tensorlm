import os
from time import time

import numpy as np
import tensorflow as tf

from tensorlm.common.util import restore_possible
from tensorlm.dataset import Vocabulary, DatasetIterator
from tensorlm.model import GeneratingLSTM
from tensorlm.trainlog import TrainState


class _BaseLM:
    def __init__(self, level, tf_session, train_text_path, max_vocab_size, neurons_per_layer,
                 num_layers, max_batch_size, num_timesteps, save_dir):
        self.train_text_path = train_text_path
        self.max_batch_size = max_batch_size
        self.num_timesteps = num_timesteps
        self.save_dir = save_dir

        if self.save_dir and not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # Build the vocabulary to determine the actual vocabulary size
        self.vocab = Vocabulary.load_or_create(save_dir, train_text_path, max_vocab_size, level)
        if self.save_dir:
            self.vocab.save_to_dir(save_dir)

        # Reload or Create a new TF model
        self.tf_model = GeneratingLSTM(vocab_size=self.vocab.get_size(),
                                       neurons_per_layer=neurons_per_layer, num_layers=num_layers,
                                       max_batch_size=max_batch_size)

        if save_dir and restore_possible(save_dir):
            ckpt = tf.train.latest_checkpoint(save_dir)
            self.tf_model.saver.restore(tf_session, ckpt)
        else:
            tf_session.run(tf.global_variables_initializer())

    def train(self, tf_session, max_epochs=10, max_steps=None, batch_size=None, text_path=None,
              log_interval=None, print_logs=False, evaluate_interval=None, evaluate_text_path=None,
              sample_interval=None, sample_prime="The ", save_interval_hours=1):
        if not text_path:
            text_path = self.train_text_path

        # Clip the batch size
        if not batch_size or batch_size > self.max_batch_size:
            batch_size = self.max_batch_size

        train_state = TrainState.try_load_from_dir(self.save_dir)
        last_save_time = time()

        while (train_state.epoch <= max_epochs and
                   (train_state.global_step <= max_steps or not max_steps)):

            # Do an epoch
            for inputs, targets in DatasetIterator(text_path, self.vocab, batch_size,
                                                   self.num_timesteps):
                loss = self.tf_model.train_step(tf_session, inputs, targets)
                train_state.train_step_done(loss, log_interval=log_interval, print_log=print_logs)

                # Evaluate
                if evaluate_interval and train_state.global_step % evaluate_interval == 0:
                    dev_loss = self.evaluate(tf_session, evaluate_text_path)
                    train_state.log_dev_loss(dev_loss, print_log=print_logs)

                # Sample
                if sample_interval and train_state.global_step % sample_interval == 0:
                    sampled = sample_prime + self.sample(tf_session, sample_prime)
                    train_state.log_sampled(sampled, print_log=print_logs)

                # Save the model and trainstate
                if (self.save_dir and save_interval_hours and
                                time() - last_save_time > save_interval_hours * 3600):
                    last_save_time = time()
                    self.tf_model.save(tf_session, self.save_dir)
                    train_state.save_to_dir(self.save_dir)

                # Check break conditions within the epoch
                if max_steps and train_state.global_step > max_steps:
                    break

            train_state.epoch_done()

    def evaluate(self, tf_session, text_path):
        dataset_iter = DatasetIterator(text_path, self.vocab, batch_size=1,
                                       num_timesteps=self.num_timesteps)
        loss = self.tf_model.evaluate(tf_session, dataset_iter)
        return loss

    def sample(self, tf_session, prime):
        return self.tf_model.sample(tf_session, self.vocab, prime)


class CharLM(_BaseLM):
    def __init__(self, tf_session, train_text_path, max_vocab_size=96, neurons_per_layer=100,
                 num_layers=3, max_batch_size=10, num_timesteps=15, save_dir=None):
        super().__init__(level="char", tf_session=tf_session, train_text_path=train_text_path,
                         max_vocab_size=max_vocab_size, neurons_per_layer=neurons_per_layer,
                         num_layers=num_layers, max_batch_size=max_batch_size,
                         num_timesteps=num_timesteps, save_dir=save_dir)


class WordLM(_BaseLM):
    def __init__(self, tf_session, train_text_path, max_vocab_size=2000, neurons_per_layer=100,
                 num_layers=2, max_batch_size=10, num_timesteps=5, save_dir=None):
        super().__init__(level="word", tf_session=tf_session, train_text_path=train_text_path,
                         max_vocab_size=max_vocab_size, neurons_per_layer=neurons_per_layer,
                         num_layers=num_layers, max_batch_size=max_batch_size,
                         num_timesteps=num_timesteps, save_dir=save_dir)
