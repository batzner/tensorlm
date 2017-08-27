# Copyright 2017 Kilian Batzner All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""TensorFlow model of an LSTM for sequence generation."""
import os

import numpy as np
import tensorflow as tf

from tensorlm.common.lstm_util import get_state_variables_for_batch, \
    get_state_update_op, get_state_variables, get_state_reset_op
from tensorlm.common.tokens import PAD_TOKEN
from tensorlm.dataset import tokenize

# Prefix of the files stored by tf.train.Saver
MODEL_FILE_PREFIX = "model"

# We distinguish between the learned model variables and the variables that store the current state
MODEL_SCOPE_NAME = "model"
LSTM_STATE_SCOPE_NAME = "lstm_state"


class GeneratingLSTM:
    """"""

    def __init__(self, vocab_size, neurons_per_layer, num_layers, max_batch_size,
                 output_keep_prob=0.5, max_gradient_norm=5,
                 initial_learning_rate=0.001, forward_only=False):
        """Creates a new LSTM for sequence generation.

        This constructor builds the computational graph of the LSTM. The variables need to be
        initialized afterwards, for example with the tf.global_variables_initializer function or by
        loading the values from a saved TF model file.

        Args:
          vocab_size (int): The input vector size
          neurons_per_layer (int): The number of units in each LSTM cell / layer
          num_layers (int): The number of LSTM cells / layers
          max_batch_size (int): The number of batches, for which the computational graph will be created.
            You can also feed a lower number (1 to max_batch_size) of batches during training or
            sampling. The computational graph's memory footprint grows linearly with this value.
          output_keep_prob (float): The probability of keeping the output for each neuron, i.e.
            1 - dropout_probability. This will only be used during training. During testing, all
            neurons keep their output.
          max_gradient_norm (float): The maximum L2 norm of the gradient during back-propagation.
          initial_learning_rate (float): The initial learning rate for the RMSProp optimization.
          forward_only (bool): If True, the graph will only be built for forward propagation. Use
            this for already trained models, whose parameters are loaded from a saved file. Setting
            this value to True will reduce the computational graph's memory footprint to about 50%.
        """
        self.neurons_per_layer = neurons_per_layer
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.max_batch_size = max_batch_size
        self.max_gradient_norm = max_gradient_norm
        self.output_keep_prob = output_keep_prob
        self.output_keep_var = tf.Variable(self.output_keep_prob, trainable=False,
                                           name="output_keep", dtype=tf.float32)
        self.learning_rate = tf.Variable(initial_learning_rate, trainable=False, name="lr",
                                         dtype=tf.float32)

        # This will be increased with each run of self.optimizer and can be used when saving the
        # model with self.saver.save(session, save_path, global_step=self.global_step)
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        # Initialize all model variables with the Xavier Initializer
        initializer = tf.contrib.layers.xavier_initializer()
        with tf.variable_scope(MODEL_SCOPE_NAME, initializer=initializer):
            self._build_graph(forward_only)

        # Define a saver for all variables that have been defined so far. Don't save the current
        # LSTM state.
        saved_variables = [v for v in tf.global_variables()
                           if not v.name.startswith(LSTM_STATE_SCOPE_NAME)]
        self.saver = tf.train.Saver(saved_variables, max_to_keep=3)

    def train_step(self, session, inputs, targets, update_state=True):
        # Returns the output tokens for each batch as a 2D ndarray and the loss
        feed_dict = {self.inputs: inputs, self.targets: targets}

        ops_without_return = [self.optimize]
        if update_state:
            ops_without_return.append(self.update_state_op)

        runs = [self.loss, ops_without_return]
        loss, _ = session.run(runs, feed_dict=feed_dict)

        return loss

    def evaluate(self, session, dataset):
        # Disable dropout and save the LSTM state before overwriting it with sampling
        self._on_pause_training(session)

        # Test the performance on the validation dataset
        total_loss = 0
        step_count = 0

        for batch_inputs, batch_targets in dataset:
            feed_dict = {self.inputs: batch_inputs, self.targets: batch_targets}
            loss, _ = session.run([self.loss, self.update_state_op], feed_dict=feed_dict)
            total_loss += loss
            step_count += 1

        # Re-enable dropout and restore the LSTM training state
        self._on_resume_training(session)
        return total_loss / step_count

    def sample(self, session, vocabulary, prime, num_steps=100):
        # Disable dropout and save the LSTM state before overwriting it with sampling
        self._on_pause_training(session)

        # Sample from the model
        prime_tokens = tokenize(prime, level=vocabulary.level)
        prime_ids = vocabulary.tokens_to_ids(prime_tokens)

        # Prime the model by feeding given inputs while only caring about its last output
        output = self._sample_step(session, np.array([prime_ids]))[0, -1]
        outputs = [output]

        # Feed the model its own output #humancentipede
        for _ in range(num_steps):
            output = self._sample_step(session, np.array([[output]]))[0, 0]
            outputs.append(output)

            # If the model output _PAD, abort
            if output == vocabulary.token_to_id[PAD_TOKEN]:
                break

        output_chars = vocabulary.ids_to_tokens(outputs)

        # Re-enable dropout and restore the LSTM training state
        self._on_resume_training(session)
        return ''.join(output_chars)

    def reset_state(self, session):
        session.run(self.reset_state_op)

    def save(self, session, save_dir):
        save_path = os.path.join(save_dir, MODEL_FILE_PREFIX)
        self.saver.save(session, save_path, global_step=self.global_step)

    def _build_graph(self, forward_only):
        # Build the central LSTM
        def layer():
            # See https://stackoverflow.com/a/44882273/2628369
            cell = tf.contrib.rnn.LSTMCell(self.neurons_per_layer)
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.output_keep_var)
            return cell

        self.cell = tf.contrib.rnn.MultiRNNCell([layer() for _ in range(self.num_layers)])

        self.logits = self._build_prediction()

        self.targets = tf.placeholder(tf.int32, [None, None])
        self.loss = self._build_loss()
        if not forward_only:
            self.optimize = self._build_optimizer()

    def _build_prediction(self):
        # Build the graph
        self.inputs = tf.placeholder(tf.int32, [None, None])
        inputs_one_hot = tf.one_hot(self.inputs, self.vocab_size)
        # inputs_one_hot will have shape (batch_size, num_timesteps, vocab_size)
        batch_size, num_timesteps = tf.shape(self.inputs)[0], tf.shape(self.inputs)[1]

        # For each layer, get the initial state. self.state will be a tuple of LSTMStateTuples. Get
        # the variables in their own scope so that, we can exclude them from being saved.
        with tf.variable_scope("lstm_state"):
            state = get_state_variables(self.cell, self.max_batch_size)

        # Unroll the LSTM
        initial_state = get_state_variables_for_batch(state, batch_size)
        outputs, new_state = tf.nn.dynamic_rnn(self.cell, inputs_one_hot,
                                               initial_state=initial_state)

        # Only get the outputs of the used batches
        outputs = outputs[:batch_size]

        # Add an operation to update the states with the last state tensors
        self.update_state_op = get_state_update_op(state, new_state)
        # Add an operation to reset the states to zero
        self.reset_state_op = get_state_reset_op(state, self.cell,
                                                 self.max_batch_size)
        # Add operations to freeze and unfreeze the state
        with tf.variable_scope(LSTM_STATE_SCOPE_NAME):
            state_frozen = get_state_variables(self.cell, self.max_batch_size)
            self.freeze_state_op = get_state_update_op(state_frozen, state)
            self.unfreeze_state_op = get_state_update_op(state, state_frozen)

        # Softmax_w is 2D, but the outputs are 3D (batch_size x num_timesteps x neurons_per_layer),
        # so we have to flatten the outputs for matrix multiplication. We merge the first two
        # dimensions and unpack them later again
        flat_outputs = tf.reshape(outputs, [-1, self.neurons_per_layer])

        softmax_w = tf.get_variable("softmax_w", [self.neurons_per_layer, self.vocab_size],
                                    initializer=tf.truncated_normal_initializer(stddev=0.1))
        softmax_b = tf.get_variable("softmax_b", [self.vocab_size],
                                    initializer=tf.constant_initializer(0.1))
        flat_logits = tf.matmul(flat_outputs, softmax_w) + softmax_b
        logits = tf.reshape(flat_logits, [-1, num_timesteps, self.vocab_size])

        return logits

    def _build_loss(self):
        # Compute cross entropy for each token.
        # logits will have shape (batch_size, timesteps, vocab_size), ao targets should have shape
        # (batch_size, timesteps).
        return tf.contrib.seq2seq.sequence_loss(
            self.logits,
            self.targets,
            tf.ones(tf.shape(self.targets)))

    def _build_optimizer(self):
        # Clip the gradients and optimize the variables
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.max_gradient_norm)
        optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        return optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)

    def _on_pause_training(self, session):
        # Disable dropout and save the current state.
        session.run([self.output_keep_var.assign(1), self.freeze_state_op])
        # Reset the state for sampling
        self.reset_state(session)

    def _on_resume_training(self, session):
        # Re-enable dropout and return to the previous training state
        session.run([self.output_keep_var.assign(self.output_keep_prob), self.unfreeze_state_op])

    def _sample_step(self, session, inputs, update_state=True):
        # Returns the LSTM"s output
        feed_dict = {self.inputs: inputs}
        runs = [self.logits]
        runs.append(self.update_state_op if update_state else tf.no_op())

        logits, _ = session.run(runs, feed_dict=feed_dict)
        return np.argmax(logits, axis=2)
