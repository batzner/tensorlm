vocabulary_size = 10000
num_layers = 3
num_neurons = 250

num_first_layer = 4 * (num_neurons * (vocabulary_size + num_neurons) + num_neurons)
num_other_layer = 4 * (num_neurons * 2 * num_neurons + num_neurons)
num_softmax = vocabulary_size * num_neurons + vocabulary_size

total = num_first_layer + (num_layers - 1) * num_other_layer + num_softmax

print("{} MM model parameters".format(total / pow(10, 6)))
