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
"""Helper functions when dealing with LSTMs in TensorFlow."""

import tensorflow as tf


def get_state_variables(cell, max_batch_size):
    """Returns TF Variables for storing the cell and hidden states of an LSTM.

    A stateful LSTM keeps its hidden and cell states after a forward propagation run and uses the
    stored states in the next forward propagation run independently of the number of steps the
    LSTM is unrolled for. For implementing such an LSTM in TF, you need to store the state in
    variables. This function returns the state variables of an LSTM cell.

    A comprehensive case for building a multilayer LSTM with a dynamic batch size and number of time
    steps is given below:

    inputs_one_hot = tf.one_hot(inputs, self.vocab_size)
    # inputs_one_hot will have shape (batch_size, num_timesteps, vocab_size)
    batch_size, num_timesteps = tf.shape(inputs_one_hot)[0], tf.shape(inputs_one_hot)[1]

    # For each layer, get the initial state. state will be a tuple of LSTMStateTuples. Get
    # the variables in their own scope so that, we can exclude them from being saved.
    with tf.variable_scope("lstm_state"):
        state = get_state_variables(cell, max_batch_size)

    # Unroll the LSTM
    initial_state = get_state_variables_for_batch(state, batch_size)
    rnn_outputs, new_state = tf.nn.dynamic_rnn(self._cell, inputs_one_hot,
                                               initial_state=initial_state)

    # Only get the outputs of the used batches
    rnn_outputs = rnn_outputs[:batch_size]

    # Add an operation to update the states with the last state tensors
    self._update_state_op = get_state_update_op(state, new_state)

    Args:
        cell (tf.contrib.rnn.MultiRNNCell): An MultiRNNCell consisting of multiple LSTMCells.
        max_batch_size (int): The maximum size of batches that are be fed to the LSTMCell. Each
            state variable in the result tuple will have max_batch_size rows in the first dimension.

    Returns:
        tuple[tf.contrib.rnn.LSTMStateTuple]: A tuple of LSTMStateTuples. Each of those will contain
            a zero cell and zero hidden state, each of shape [max_batch_size, state_size]. The
            length of the result tuple will be determined by the cell's number of layers (e.g. for
            an MultiRNNCell).
    """

    # For each layer, get the initial state tuple and make two variables out of it to enable setting
    # its value.
    state_variables = []
    for state_c, state_h in cell.zero_state(max_batch_size, tf.float32):
        state_variables.append(tf.contrib.rnn.LSTMStateTuple(
            tf.Variable(state_c, trainable=False),
            tf.Variable(state_h, trainable=False)))
    return tuple(state_variables)


def get_state_variables_for_batch(state_variables, batch_size):
    """Returns a subset of the state variables.

    This function takes the state variables returned by get_state_variables() and returns a subset
    for an actual forward-propagation run. Specifically, it clips each of the state variables to
    the given batch size.

    Before this call, each variable's first dimension has length
    max_batch_size but when the input has a lower batch size, the LSTM should also only update the
    state variables for the used batches.

    See get_state_variables() for more info.

    Args:
        state_variables (tuple[tf.contrib.rnn.LSTMStateTuple]): The LSTM's state variables.
        batch_size (tf.Tensor): An 0-dimensional tensor containing the batch size tensor in the
            computational graph.

    Returns:
        tuple[tf.contrib.rnn.LSTMStateTuple]: A new tuple of state variables clipped to the given
            batch size.
    """

    # Return a tuple of LSTMStateTuples but with only the first batch_size rows for each variable
    # in the tuples.
    result = []
    for state_c, state_h in state_variables:
        result.append(tf.contrib.rnn.LSTMStateTuple(state_c[:batch_size], state_h[:batch_size]))
    return tuple(result)


def get_state_update_op(state_variables, new_states):
    """Returns an operation to update an LSTM's state variables.

    See get_state_variables() for more info.

    Args:
        state_variables (tuple[tf.contrib.rnn.LSTMStateTuple]): The LSTM's state variables.
        new_states (tuple[tf.contrib.rnn.LSTMStateTuple]): The new values for the state variables.
            new_states may have state tuples with state sizes < max_batch_size. Then, only the first
            rows of the corresponding state variables will be updated.

    Returns:
        tf.Operation: An operation that updates the LSTM's.
    """

    # Add an operation to update the train states with the last state tensors.
    update_ops = []
    for state_variable, new_state in zip(state_variables, new_states):
        # new_state[0] might be smaller than state_variable[0], because state_variable[0]
        # contains max_batch_size entries.

        # Get the update indices for both states in the tuple
        update_indices = (tf.range(0, tf.shape(new_state[0])[0]),
                          tf.range(0, tf.shape(new_state[1])[0]))
        update_ops.extend([
            tf.scatter_update(state_variable[0], update_indices[0], new_state[0]),
            tf.scatter_update(state_variable[1], update_indices[1], new_state[1])
        ])
    return tf.tuple(update_ops)


def get_state_reset_op(state_variables, cell, max_batch_size):
    """Returns an operation to set each variable in a list of LSTMStateTuples to zero.

    See get_state_variables() for more info.

    Args:
        state_variables (tuple[tf.contrib.rnn.LSTMStateTuple]): The LSTM's state variables.
        cell (tf.contrib.rnn.MuliRNNCell): An MultiRNNCell consisting of multiple LSTMCells.
        max_batch_size (int): The maximum size of batches that are be fed to the LSTMCell.

    Returns:
        tf.Operation: An operation that sets the LSTM's state to zero.
    """
    zero_states = cell.zero_state(max_batch_size, tf.float32)
    return get_state_update_op(state_variables, zero_states)


def get_num_params(vocab_size, num_layers, num_neurons):
    """Returns the number of trainable parameters of an LSTM.

    Args:
        vocab_size (int): The vocabulary size
        num_layers (int): The number of layers in the LSTM
        num_neurons (int): The number of neurons / units per layer

    Returns:
        int: The number of trainable parameters
    """
    num_first_layer = 4 * (num_neurons * (vocab_size + num_neurons) + num_neurons)
    num_other_layer = 4 * (num_neurons * 2 * num_neurons + num_neurons)
    num_softmax = vocab_size * num_neurons + vocab_size

    return num_first_layer + (num_layers - 1) * num_other_layer + num_softmax
