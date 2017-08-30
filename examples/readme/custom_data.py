import numpy as np
import tensorflow as tf

from tensorlm import GeneratingLSTM

with tf.Session() as session:
    # We use integer ids from 0 to 19, so the vocab size is 20. The range of ids must always start
    # at zero.
    batch_inputs = np.array([[1, 2, 3, 4], [15, 16, 17, 18]])  # 2 batches, 4 time steps each
    batch_targets = np.array([[2, 3, 4, 5], [16, 17, 18, 19]])

    # Create the model in a TensorFlow graph
    model = GeneratingLSTM(vocab_size=20, neurons_per_layer=10, num_layers=2, max_batch_size=2)

    # Initialize all defined TF Variables
    session.run(tf.global_variables_initializer())

    for _ in range(5000):
        model.train_step(session, batch_inputs, batch_targets)

    sampled = model.sample_ids(session, [15], num_steps=3)
    print("Sampled: " + str(sampled))
