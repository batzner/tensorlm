import numpy as np
import tensorflow as tf

from src.generating_lstm.common.lstm_util import get_state_variables_for_batch, \
    get_state_update_op, get_state_variables, get_state_reset_op
from src.generating_lstm.common.tokens import PAD_TOKEN


class GeneratingLSTM:
    def __init__(self, vocab_size, num_neurons, num_layers, max_batch_size,
                 output_keep_prob=1, max_gradient_norm=5,
                 initial_learning_rate=0.001, forward_only=False):
        self.num_neurons = num_neurons
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.max_batch_size = max_batch_size
        self.max_gradient_norm = max_gradient_norm
        self.output_keep_prob = output_keep_prob
        self.output_keep_var = tf.Variable(output_keep_prob, trainable=False, name="output_keep")
        self.learning_rate = tf.Variable(initial_learning_rate, trainable=False, name="lr")
        self.forward_only = forward_only

        initializer = tf.contrib.layers.xavier_initializer()
        with tf.variable_scope("model", initializer=initializer):
            self._build_graph()

        saved_variables = [v for v in tf.global_variables() if not v.name.startswith("lstm_state")]
        self.saver = tf.train.Saver(saved_variables, max_to_keep=1)

    def train_step(self, session, inputs, targets, update_state=True):
        # Returns the output tokens for each batch as a 2D ndarray and the loss
        feed_dict = {self.inputs: inputs, self.targets: targets}

        ops_without_return = [self.optimize]
        if update_state:
            ops_without_return.append(self.update_state_op)

        runs = [self.loss, ops_without_return]
        loss, _ = session.run(runs, feed_dict=feed_dict)

        return loss

    def evaluate(self, session, dataset, batch_size, num_timesteps):
        # Disable dropout and save the LSTM state before overwriting it with sampling
        self._on_pause_training(session)

        # Test the performance on the validation dataset
        total_loss = 0
        step_count = 0
        for batch_inputs, batch_targets in dataset.get_batch_iter(batch_size, num_timesteps):
            feed_dict = {self.inputs: batch_inputs, self.targets: batch_targets}
            loss, _ = session.run([self.loss, self.update_state_op], feed_dict=feed_dict)
            total_loss += loss
            step_count += 1

        # Re-enable dropout and restore the LSTM training state
        self._on_resume_training(session)
        return total_loss / step_count

    def sample(self, session, dataset, prime, steps=100, stop_tokens=None, max_stops=1):
        # Disable dropout and save the LSTM state before overwriting it with sampling
        self._on_pause_training(session)

        stop_ids = []
        if stop_tokens:
            stop_ids = [dataset.token_to_id(w) for w in stop_tokens]

        # Sample from the model
        prime_tokens = dataset.tokenize_to_ids(prime)

        # Prime the model by feeding given inputs while only caring about its last output
        output = self._sample_step(session, np.array([prime_tokens]))[0, -1]
        outputs = [output]

        # Feed the model its own output #humancentipede
        seen_stops = 0
        for _ in range(steps):
            output = self._sample_step(session, np.array([[output]]))[0, 0]
            outputs.append(output)

            # If the model output _PAD, abort
            if output == dataset.token_to_id(PAD_TOKEN):
                break

            # Check, if the model output a stop token
            if output in stop_ids:
                seen_stops += 1
                if seen_stops >= max_stops:
                    break

        output_text = dataset.token_ids_to_sentence(outputs)

        # Re-enable dropout and restore the LSTM training state
        self._on_resume_training(session)
        return output_text

    def reset_state(self, session):
        session.run(self.reset_state_op)

    def _build_graph(self):
        # Build the central LSTM
        cell = tf.nn.rnn_cell.LSTMCell(self.num_neurons)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.output_keep_var)
        self.cell = tf.nn.rnn_cell.MultiRNNCell([cell] * self.num_layers)

        self.logits, flat_logits = self._build_prediction()

        self.targets = tf.placeholder(tf.int32, [None, None])
        self.loss = self._build_loss(flat_logits)
        if not self.forward_only:
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
        with tf.variable_scope("lstm_state"):
            state_frozen = get_state_variables(self.cell, self.max_batch_size)
            self.freeze_state_op = get_state_update_op(state_frozen, state)
            self.unfreeze_state_op = get_state_update_op(state, state_frozen)

        # Softmax_w is 2D, but the outputs are 3D (batch_size x num_timesteps x num_neurons),
        # so we have to flatten the outputs for matrix multiplication. We merge the first two
        # dimensions and unpack them later again
        flat_outputs = tf.reshape(outputs, [-1, self.num_neurons])

        softmax_w = tf.get_variable("softmax_w", [self.num_neurons, self.vocab_size],
                                    initializer=tf.truncated_normal_initializer(stddev=0.1))
        softmax_b = tf.get_variable("softmax_b", [self.vocab_size],
                                    initializer=tf.constant_initializer(0.1))
        flat_logits = tf.matmul(flat_outputs, softmax_w) + softmax_b
        logits = tf.reshape(flat_logits, [-1, num_timesteps, self.vocab_size])

        return logits, flat_logits

    def _build_loss(self, flat_logits):
        # Compute cross entropy for each token.
        # logits will have shape (batch_size*timesteps, vocab_size), so targets should have shape
        # (batch_size*timesteps).
        flat_targets = tf.reshape(self.targets, [-1])
        return tf.nn.seq2seq.sequence_loss(
            [flat_logits],
            [flat_targets],
            [tf.ones(tf.shape(flat_targets))])

    def _build_optimizer(self):
        # Clip the gradients and optimize the variables
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.max_gradient_norm)
        optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        return optimizer.apply_gradients(zip(grads, tvars))

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
