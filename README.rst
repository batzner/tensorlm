tensorlm
========

Generate Shakespeare poems with 4 lines of code.

<a href="http://www.mlowl.com/post/character-language-model-lstm-tensorflow/" target="_blank">[![showcase
of the package]]</a>

Installation
------------

`tensorlm` is written in / for Python 3.

    pip3 install tensorflow>=1.1
    pip3 install tensorlm

Basic Usage
-----------

Use the `CharLM` or `WordLM` class:

``` {.python}
import tensorflow as tf
from tensorlm import CharLM

with tf.Session() as session:

    # Create a new model. You can also use WordLM
    model = CharLM(session, "datasets/sherlock/train.txt", max_vocab_size=96,
                   neurons_per_layer=100, num_layers=3, num_timesteps=15)

    # Train it
    model.train(session, max_epochs=5, max_steps=500, print_logs=True)

    # Let it generate a text
    generated = model.sample(session, "The ", num_steps=100)
    print("The " + generated)
```

This should output something like:

    The     eee       ee      ee      ee     e e     ee      ee     e  e    e  e    e  e    e  e    e  e

Command Line Usage
------------------

**Train:**\
`python3 -m tensorlm.cli --train=True --level=char --train_text_path=datasets/sherlock/train.txt --max_vocab_size=96 --neurons_per_layer=100 --num_layers=3 --batch_size=10 --num_timesteps=160 --save_dir=out/model --max_epochs=300 --save_interval_hours=0.5`

**Sample:**\
`python3 -m tensorlm.cli --sample=True --level=char --neurons_per_layer=400 --num_layers=3 --num_timesteps=160 --save_dir=out/model`

**Evaluate:**\
`python3 -m tensorlm.cli --evaluate=True --level=char --evaluate_text_path=datasets/sherlock/valid.txt --neurons_per_layer=400 --num_layers=3 --batch_size=10 --num_timesteps=160 --save_dir=out/model`

See `python3 -m tensorlm.cli --help` for all options.

Advanced Usage
--------------

### Custom Input Data

The inputs and targets don’t have to be text. `GeneratingLSTM` only
expects token ids, so you can use any data type for the sequences, as
long as you can encode the data to integer ids.

``` {.python}
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
```

This should output something like:

    Sampled: [16, 18, 19]

### Custom Training, Dropout etc.

Use the `GeneratingLSTM` class directly. This class is agnostic to the
dataset type. It expects integer ids and returns integer ids.

\`\`\`python\
import tensorflow

  [showcase of the package]: http://i.cubeupload.com/8Cm5RQ.gif
  [![showcase of the package]]: http://www.mlowl.com/post/character-language-model-lstm-tensorflow/