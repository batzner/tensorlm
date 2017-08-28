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
"""TensorFlow model of an LSTM for sequence generation."""

import numpy as np
import tensorflow as tf

from tensorlm.common.lstm_util import get_state_variables_for_batch, \
    get_state_update_op, get_state_variables, get_state_reset_op
from tensorlm.common.tokens import PAD_TOKEN
from tensorlm.dataset import tokenize

# We distinguish between the learned model variables and the variables that store the current state
MODEL_SCOPE_NAME = "model"
LSTM_STATE_SCOPE_NAME = "lstm_state"


class GeneratingLSTM:
    """TensorFlow model of an LSTM for sequence generation.

    Use this class to train an LSTM model. This class does all the TensorFlow work for you, so you
    can use it for training, evaluating and sampling by just feeding it numpy arrays. The model
    itself is not text-aware, it just accepts token ids (integers) and returns a probability
    distribution for the output token ids.

    Use tensorlm.dataset.Vocabulary for translating the ids to tokens.

    To prevent the model from forgetting its memory state between train_steps, we store the LSTM's
    cell and hidden state. So, you can feed a long text chunk by chunk into the model and it updates
    its state after each feed. You can reset the LSTM's cell and hidden state by calling
    reset_state().

    For training and evaluation you can use varying batch sizes and number of time steps, even
    between function calls to train_step(). The number of time steps denotes the number of steps
    that the LSTM is unrolled for. See tf.nn.dynamic_rnn for more info. The batch size and number of
    time steps are determined dynamically based on the input size.

    For training, RMSProp is used.

    For saving / reloading the model to / from the filesystem, use the self.saver member of the
    instance. This will be a tf.train.Saver. See the TensorFlow documentation for info about how to
    use the tf.train.Saver. The LSTM's cell and hidden state won't be saved in the output file. Only
    trainable parameters will be saved.
    """

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
            max_batch_size (int): The number of batches, for which the computational graph will be
                created. You can also feed a lower number (1 to max_batch_size) of batches during
                training or sampling. The computational graph's memory footprint grows linearly with
                this value.
            output_keep_prob (float): The probability of keeping the output for each neuron, i.e.
                1 - dropout_probability. This will only be used during training. During testing, all
                neurons keep their output.
            max_gradient_norm (float): The maximum L2 norm of the gradient during back-propagation.
            initial_learning_rate (float): The initial learning rate for the RMSProp optimization.
            forward_only (bool): If True, the graph will only be built for forward propagation. Use
                this for already trained models, whose parameters are loaded from a saved file.
                Setting this value to True will reduce the computational graph's memory footprint
                to about 50%.
        """
        self.neurons_per_layer = neurons_per_layer
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.max_batch_size = max_batch_size
        self.max_gradient_norm = max_gradient_norm
        self.output_keep_prob = output_keep_prob
        self._output_keep_var = tf.Variable(self.output_keep_prob, trainable=False,
                                            name="output_keep", dtype=tf.float32)
        self._learning_rate = tf.Variable(initial_learning_rate, trainable=False, name="lr",
                                          dtype=tf.float32)

        # Before building the graph, define the root placeholders
        self._inputs = tf.placeholder(tf.int32, [None, None])
        self._targets = tf.placeholder(tf.int32, [None, None])

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
        """Runs one training step / back-propagation run and returns the loss.

        This method feeds the inputs and targets into the computational graph, determines the loss
        and updates the model's parameters based on the loss with back-propagation.

        The batch size and number of timesteps, for which the LSTM is unrolled, will be determined

        Args:
            session (tf.Session): The TF session to run the operations in.
            inputs (np.ndarray): A batch of training inputs. Must have the shape
                (batch_size, num_timesteps) and contain only integers.
            targets (np.ndarray): A batch of training targets. Must have the shape
                (batch_size, num_timesteps) and contain only integers. For an input batch of
                [[1,2,3,4]], the targets should be [[2,3,4,?]].
            update_state (bool): If True, this will update the LSTM's state / memory.

        Returns:
            float: The mean cross-entropy loss on this batch.
        """

        # Returns the output tokens for each batch as a 2D ndarray and the loss
        feed_dict = {self._inputs: inputs, self._targets: targets}

        ops_without_return = [self._optimize]
        if update_state:
            ops_without_return.append(self._update_state_op)

        runs = [self._loss, ops_without_return]
        loss, _ = session.run(runs, feed_dict=feed_dict)

        return loss

    def evaluate(self, session, dataset):
        """Determines the mean cross-entropy loss on a dataset.

        Feeds a given dataset in batches to the model and compares the model's outputs with the
        dataset's targets using the cross-entropy loss. The batch size and number of timesteps fed
        at once are given by the dataset.

        Args:
            session (tf.Session): The TF session to run the operations in.
            dataset (tensorlm.dataset.Dataset): A dataset to provide the batch inputs and batch
                targets.

        Returns:
            float: The mean cross-entropy loss on the dataset.
        """

        # Disable dropout and save the LSTM state before overwriting it with sampling
        self._on_pause_training(session)

        # Test the performance on the validation dataset
        total_loss = 0
        step_count = 0

        for batch_inputs, batch_targets in dataset:
            feed_dict = {self._inputs: batch_inputs, self._targets: batch_targets}
            loss, _ = session.run([self._loss, self._update_state_op], feed_dict=feed_dict)
            total_loss += loss
            step_count += 1

        # Re-enable dropout and restore the LSTM training state
        self._on_resume_training(session)
        return total_loss / step_count

    def sample(self, session, vocabulary, prime, num_steps=100):
        """Let the model generate a sequence based on a preceding string.

        This method tokenizes the prime string and feeds the tokens to the model. Then, it feeds the
        model its own output (disgusting, I know) token by token and thus lets it generate /
        complete the text. For char level, this will result in 100 generated characters, for word
        level 100 generated tokens (words / punctuation / whitespace).

        Args:
            session (tf.Session): The TF session to run the operations in.
            vocabulary (tf.dataset.Vocabulary): A vocabulary for tokenizing the prime string and
                translating the output ids back to tokens.
            prime (str): A string to prime the model with.
            num_steps (int): The number of tokens generated by the model.

        Returns:
            str: The generated text.
        """
        # Disable dropout and save the LSTM state before overwriting it with sampling
        self._on_pause_training(session)

        # Sample from the model
        prime_tokens = tokenize(prime, level=vocabulary.level)
        prime_ids = vocabulary.tokens_to_ids(prime_tokens)

        # Prime the model by feeding given inputs while only caring about its last output
        output = self._sample_step(session, np.array([prime_ids]))[0, -1]
        outputs = [output]

        # Feed the model its own output #humancentipede
        for _ in range(num_steps - 1):
            # Feed one batch with one timestep
            batch_input = np.array([[output]])
            batch_output = self._sample_step(session, batch_input)
            output = batch_output[0, -1]
            outputs.append(output)

            # If the model output _PAD, abort
            if output == vocabulary.token_to_id[PAD_TOKEN]:
                break

        output_tokens = vocabulary.ids_to_tokens(outputs)

        # Re-enable dropout and restore the LSTM training state
        self._on_resume_training(session)
        return ''.join(output_tokens)

    def reset_state(self, session):
        """Resets the LSTM's memory state to zero.

        Args:
            session (tf.Session): The TF session to run the operations in.
        """
        session.run(self._reset_state_op)

    def _build_graph(self, forward_only):
        """Builds the whole computational graph for the LSTM.

        Args:
            forward_only (bool): If True, the graph will also contain back-propagation operations
                for improving the model's trainable parameters.
        """

        self._cell = tf.contrib.rnn.MultiRNNCell(
            [self._build_lstm_layer() for _ in range(self.num_layers)])
        self._logits = self._build_prediction()
        self._loss = self._build_loss()

        if not forward_only:
            self._optimize = self._build_optimizer()

    def _build_lstm_layer(self):
        """Returns a dropout-wrapped LSTM-cell.

        See https://stackoverflow.com/a/44882273/2628369 for why this local function is necessary.

        Returns:
            tf.contrib.rnn.DropoutWrapper: The dropout-wrapped LSTM cell.
        """
        cell = tf.contrib.rnn.LSTMCell(self.neurons_per_layer)
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self._output_keep_var)
        return cell

    def _build_prediction(self):
        """Builds the forward-propagation part of the computational graph.

        Returns:
            tf.Tensor: A tensor of shape (batch_size, num_timesteps, self.vocab_size) containing the
                log-probability for each output token. Note: the nested vectors of length
                self.vocab_size will have a sum of 1. They would be put into a softmax function to
                get 1-sum-vectors. However, with the argmax function, you can transform each vector
                to the model's predicted token.
        """
        # Build the forward propagation graph
        inputs_one_hot = tf.one_hot(self._inputs, self.vocab_size)
        # inputs_one_hot will have shape (batch_size, num_timesteps, vocab_size)
        batch_size, num_timesteps = tf.shape(self._inputs)[0], tf.shape(self._inputs)[1]

        # For each layer, get the initial state. self.state will be a tuple of LSTMStateTuples. Get
        # the variables in their own scope so that, we can exclude them from being saved.
        with tf.variable_scope("lstm_state"):
            state = get_state_variables(self._cell, self.max_batch_size)

        # Unroll the LSTM
        initial_state = get_state_variables_for_batch(state, batch_size)
        rnn_outputs, new_state = tf.nn.dynamic_rnn(self._cell, inputs_one_hot,
                                                   initial_state=initial_state)

        # Only get the outputs of the used batches
        rnn_outputs = rnn_outputs[:batch_size]

        # Add an operation to update the states with the last state tensors
        self._update_state_op = get_state_update_op(state, new_state)
        # Add an operation to reset the states to zero
        self._reset_state_op = get_state_reset_op(state, self._cell,
                                                  self.max_batch_size)
        # Add operations to freeze and unfreeze the state
        with tf.variable_scope(LSTM_STATE_SCOPE_NAME):
            state_frozen = get_state_variables(self._cell, self.max_batch_size)
            self._freeze_state_op = get_state_update_op(state_frozen, state)
            self._unfreeze_state_op = get_state_update_op(state, state_frozen)

        # Softmax_w is 2D, but the outputs are 3D (batch_size x num_timesteps x neurons_per_layer),
        # so we have to flatten the outputs for matrix multiplication. We merge the first two
        # dimensions and unpack them later again
        flat_rnn_outputs = tf.reshape(rnn_outputs, [-1, self.neurons_per_layer])

        softmax_w = tf.get_variable("softmax_w", [self.neurons_per_layer, self.vocab_size],
                                    initializer=tf.truncated_normal_initializer(stddev=0.1))
        softmax_b = tf.get_variable("softmax_b", [self.vocab_size],
                                    initializer=tf.constant_initializer(0.1))
        flat_logits = tf.matmul(flat_rnn_outputs, softmax_w) + softmax_b
        logits = tf.reshape(flat_logits, [-1, num_timesteps, self.vocab_size])

        return logits

    def _build_loss(self):
        """Build the part of the graph to compute the mean cross entropy loss for the outputs

        Returns:
            tf.Tensor: A float tensor of rank 0 containing the average loss across time steps and
                batches.
        """
        # self.logits will have shape (batch_size, timesteps, vocab_size)
        # self.targets will have shape (batch_size, timesteps)
        return tf.contrib.seq2seq.sequence_loss(
            self._logits,
            self._targets,
            tf.ones(tf.shape(self._targets)))

    def _build_optimizer(self):
        """Based on the loss tensor, build an optimizer that minimizes the loss.

        This function returns an optimizer operation that updates the model's trainable parameters
        by determining the loss's gradients w.r.t. each of the trainable parameters. Specifically,
        RMSProp is used to minimize the loss. The gradients are clipped to the max_gradient_norm to
        prevent too drastic updates of the trainable parameters. See also tf.clip_by_global_norm

        Returns:
            tf.Operation: An operation that updates the model's trainable parameters.
        """

        # Clip the gradients
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self._loss, tvars), self.max_gradient_norm)

        # Optimize the variables
        optimizer = tf.train.RMSPropOptimizer(self._learning_rate)
        return optimizer.apply_gradients(zip(grads, tvars))

    def _on_pause_training(self, session):
        """Disables dropout and freezes the LSTM's current memory state.

        This function prepares the model for doing anything else during training. It disables
        dropout and stores the current memory state in a private variable. Then, it resets the
        LSTM's memory state to zero. For resuming training / unfreezing the saved memory state, see
        _on_resume_training.

        Args:
            session (tf.Session): The TF session to run the operations in.
        """
        # Disable dropout and save the current state.
        session.run([self._output_keep_var.assign(1), self._freeze_state_op])
        # Reset the state for sampling or evaluation
        self.reset_state(session)

    def _on_resume_training(self, session):
        """Enables dropout and unfreezes the saved memory state.

        See _on_pause_training for info about the saved memory state.

        Args:
            session (tf.Session): The TF session to run the operations in.
        """
        # Re-enable dropout and return to the previous training state
        session.run([self._output_keep_var.assign(self.output_keep_prob), self._unfreeze_state_op])

    def _sample_step(self, session, inputs, update_state=True):
        """Feeds batch inputs to the model and returns the batch output ids.

        Args:
            session (tf.Session): The TF session to run the operations in.
            inputs (np.ndarray): A batch of inputs. Must have the shape (batch_size, num_timesteps)
                and contain only integers. The batch size and number of timesteps are determined
                dynamically, so the shape of inputs can vary between calls of this function.
            update_state (bool): If True, this will update the LSTM's memory state. If this function
                gets called during training, make sure to call it between _on_pause_training and
                _on_resume_training.

        Returns:
            np.ndarray: A batch of outputs with the same shape and data type as the inputs
                parameter.
        """
        # Feed the input
        feed_dict = {self._inputs: inputs}
        runs = [self._logits]
        runs.append(self._update_state_op if update_state else tf.no_op())

        # Get the output
        logits, _ = session.run(runs, feed_dict=feed_dict)
        return np.argmax(logits, axis=2)
