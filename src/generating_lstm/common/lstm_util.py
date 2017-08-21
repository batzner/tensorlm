import tensorflow as tf


def get_state_variables(cell, batch_size):
    # For each layer, get the initial state tuple and make two variables out of it to enable setting
    # its value.
    state_variables = []
    for state_c, state_h in cell.zero_state(batch_size, tf.float32):
        state_variables.append(tf.contrib.rnn.LSTMStateTuple(
            tf.Variable(state_c, trainable=False),
            tf.Variable(state_h, trainable=False)))
    return tuple(state_variables)


def get_state_variables_for_batch(state_variables, batch_size):
    # Return a tuple of LSTMStateTuples but with only the first batch_size rows for each variable
    # in the tuples.
    result = []
    for state_c, state_h in state_variables:
        result.append(tf.contrib.rnn.LSTMStateTuple(state_c[:batch_size], state_h[:batch_size]))
    return tuple(result)


def get_state_update_op(state_variables, new_states):
    # Add an operation to update the train states with the last state tensors. new_states may
    # have state tuples with state sizes < max_batch_size. Then, only the first rows of the
    # corresponding state variables will be updated.
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


def get_state_reset_op(state_variables, cell, batch_size):
    # Return an operation to set each variable in a list of LSTMStateTuples to zero
    zero_states = cell.zero_state(batch_size, tf.float32)
    return get_state_update_op(state_variables, zero_states)