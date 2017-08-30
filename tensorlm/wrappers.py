# Copyright (c) 2017 Kilian Batzner All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================
"""Wrappers around tensorlm.model that only need the path to the .txt to do the whole training."""

import os
from time import time

import tensorflow as tf

from tensorlm.common.log import get_logger
from tensorlm.common.util import restore_possible
from tensorlm.common.trainlog import TrainState
from tensorlm.dataset import Vocabulary, Dataset
from tensorlm.model import GeneratingLSTM

LOGGER = get_logger(__name__)

# Prefix of the files stored by tf.train.Saver
MODEL_FILE_PREFIX = "model"


class BaseLM:
    """Wrappers around tensorlm.model that only needs the path to the .txt to do the training."""

    def __init__(self, tf_session, level, train_text_path, max_vocab_size, neurons_per_layer,
                 num_layers, batch_size, num_timesteps, save_dir):
        """Creates a new language model without training it.

        Args:
            tf_session (tf.Session): The session to run the TF Variable initializer in.
            level (str): The level for tokenizing the text - either "char" or "word".
            train_text_path (str): Path to the .txt file containing the training text
            max_vocab_size (int): Maximum size of the vocabulary to use to translate tokens into
                ids. If the text contains a lower number of distinct tokens than the max_vocab_size,
                the vocabulary will be smaller than max_vocab_size.
            neurons_per_layer (int): Number of neurons / units per layer.
            num_layers (int): Number of LSTM layers.
            batch_size (int): The batch size to use for training and evaluation.
            num_timesteps (int): The number of time steps to unroll the LSTM for. The
                back-propagation / training will only go num_timesteps steps into the past. A higher
                number makes the LSTM remember information from tokens that it read earlier.
            save_dir (str): Path to the directory, where the model's learned parameters, the
                vocabulary and training statistics will be saved. Use None, to disable saving
                anything.
        """
        self.train_text_path = train_text_path
        self._batch_size = batch_size
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
                                       max_batch_size=batch_size)

        if save_dir and restore_possible(save_dir):
            ckpt = tf.train.latest_checkpoint(save_dir)
            self.tf_model.saver.restore(tf_session, ckpt)
        else:
            tf_session.run(tf.global_variables_initializer())

    def train(self, tf_session, max_epochs=10, max_steps=None, text_path=None,
              log_interval=None, print_logs=False, evaluate_interval=None, evaluate_text_path=None,
              sample_interval=None, sample_prime="The ", save_interval_hours=1):
        """Train the language model on a text file.

        Args:
            tf_session (tf.Session): The TF Session to run the operations in.
            max_epochs (int): Maximum number of epochs to train.
            max_steps (int): (Optional) Maximum number of training steps. Training will also be
                limited by max_epochs. If max_steps is None, this parameter will be ignored.
            text_path (str): (Optional) If set, the training will not run on the text path that the
                model's vocabulary was built on in the constructor but on the specified text file.
            log_interval (int): (Optional) Number of steps between logs. If None, there will be no
                logs. If not None, logs about the training will be saved to a trainlog.json file.
            print_logs (bool): If True, logs will not only be saved to a trainlog.json but also be
                printed.
            evaluate_interval (int): (Optional) Number of steps between evaluation runs. If None,
                there will be no evaluation runs. If not None, evaluate_text_path should be set as
                well.
            evaluate_text_path (str): (Optional) Path to the text file to evaluate the model on
                without training it. If this parameter is set, evaluate_interval must also be set.
            sample_interval (int): (Optional) Number of steps between sampling from the model. If
                None, there will be no sampling during training. if not None, sample_prime should be
                set as well.
            sample_prime (str): Sequence to prime the model before each sampling run.
            save_interval_hours (float): Save the model's trainable parameters to the file system
                every x hours. The files will be in self.save_dir.
        """

        if not text_path:
            text_path = self.train_text_path

        train_state = TrainState.try_load_from_dir(self.save_dir)
        last_save_time = time()

        train_set = Dataset(text_path, self.vocab, self._batch_size, self.num_timesteps)

        while (train_state.epoch <= max_epochs and
                   (not max_steps or train_state.global_step <= max_steps)):

            batch = train_set.get_batch(train_state.step_in_epoch)

            # Check if the epoch end was reached
            if not batch:
                train_state.epoch_done()
                train_state.step_in_epoch = 0
                continue

            inputs, targets = batch
            loss = self.tf_model.train_step(tf_session, inputs, targets)

            # Increase the global step and step in epoch
            train_state.train_step_done(loss, log_interval=log_interval, print_log=print_logs)

            # Evaluate
            if evaluate_interval and train_state.global_step % evaluate_interval == 0:
                dev_loss = self.evaluate(tf_session, evaluate_text_path)
                train_state.log_dev_loss(dev_loss, print_log=print_logs)

            # Sample
            if sample_interval and train_state.global_step % sample_interval == 0:
                sampled = sample_prime + self.sample(tf_session, sample_prime, num_steps=100)
                train_state.log_sampled(sampled, print_log=print_logs)

            # Save the model and train state
            if (self.save_dir and save_interval_hours and
                            time() - last_save_time > save_interval_hours * 3600):
                last_save_time = time()
                # Save the model
                save_path = os.path.join(self.save_dir, MODEL_FILE_PREFIX)
                self.tf_model.saver.save(tf_session, save_path)
                # Save the train state
                train_state.save_to_dir(self.save_dir)

    def evaluate(self, tf_session, text_path):
        """Evaluate the model's performance on a given text path.

        Args:
            tf_session (tf.Session): The TF session to run the operations in.
            text_path (str): Path to the .txt file for evaluation.

        Returns:
            float: The average cross-entropy loss on the dataset.
        """
        dataset = Dataset(text_path, self.vocab, batch_size=self._batch_size,
                          num_timesteps=self.num_timesteps)
        loss = self.tf_model.evaluate(tf_session, dataset)
        return loss

    def sample(self, tf_session, prime, num_steps):
        """Let the model generate text after being primed with some text.

        Args:
            tf_session (tf.Session): The TF session to run the operations in.
            prime (str): The text to prime the model with.
            num_steps (int): The number of tokens to generate. For a character level language model,
                this will result in num_steps generated characters. For word level, this will result
                in num_steps words / numbers / punctuation marks / whitespace characters

        Returns:
            str: The generated sequence.
        """
        return self.tf_model.sample_text(tf_session, self.vocab, prime, num_steps)


class CharLM(BaseLM):
    def __init__(self, tf_session, train_text_path, max_vocab_size=96, neurons_per_layer=100,
                 num_layers=3, batch_size=10, num_timesteps=15, save_dir=None):
        """Creates a new character-level language model without training it.

        Args:
            tf_session (tf.Session): The session to run the TF Variable initializer in.
            train_text_path (str): Path to the .txt file containing the training text
            max_vocab_size (int): Maximum size of the vocabulary to use to translate tokens into
                ids. If the text contains a lower number of distinct tokens than the max_vocab_size,
                the vocabulary will be smaller than max_vocab_size.
            neurons_per_layer (int): Number of neurons / units per layer.
            num_layers (int): Number of LSTM layers.
            batch_size (int): The batch size to use for training and evaluation.
            num_timesteps (int): The number of time steps to unroll the LSTM for. The
                back-propagation / training will only go num_timesteps steps into the past. A higher
                number makes the LSTM remember information from tokens that it read earlier.
            save_dir (str): Path to the directory, where the model's learned parameters, the
                vocabulary and training statistics will be saved. Use None, to disable saving
                anything.
        """
        super().__init__(tf_session=tf_session, level="char", train_text_path=train_text_path,
                         max_vocab_size=max_vocab_size, neurons_per_layer=neurons_per_layer,
                         num_layers=num_layers, batch_size=batch_size,
                         num_timesteps=num_timesteps, save_dir=save_dir)


class WordLM(BaseLM):
    def __init__(self, tf_session, train_text_path, max_vocab_size=2000, neurons_per_layer=100,
                 num_layers=2, batch_size=10, num_timesteps=5, save_dir=None):
        """Creates a new word-level language model without training it.

        Args:
            tf_session (tf.Session): The session to run the TF Variable initializer in.
            train_text_path (str): Path to the .txt file containing the training text
            max_vocab_size (int): Maximum size of the vocabulary to use to translate tokens into
                ids. If the text contains a lower number of distinct tokens than the max_vocab_size,
                the vocabulary will be smaller than max_vocab_size.
            neurons_per_layer (int): Number of neurons / units per layer.
            num_layers (int): Number of LSTM layers.
            batch_size (int): The batch size to use for training and evaluation.
            num_timesteps (int): The number of time steps to unroll the LSTM for. The
                back-propagation / training will only go num_timesteps steps into the past. A higher
                number makes the LSTM remember information from tokens that it read earlier.
            save_dir (str): Path to the directory, where the model's learned parameters, the
                vocabulary and training statistics will be saved. Use None, to disable saving
                anything.
        """
        super().__init__(tf_session=tf_session, level="word", train_text_path=train_text_path,
                         max_vocab_size=max_vocab_size, neurons_per_layer=neurons_per_layer,
                         num_layers=num_layers, batch_size=batch_size,
                         num_timesteps=num_timesteps, save_dir=save_dir)
