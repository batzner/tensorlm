# tensorlm

Generate Shakespeare poems with 4 lines of code.

## Installation

    pip install tensorflow
    pip install tensorlm
    
## Usage

    import tensorflow as tf
    from tensorlm import WordLM
        
    with tf.Session() as session:
        
        # Create a new model
        model = CharLM(session, "datasets/sherlock/train.txt", max_vocab_size=96,
                       neurons_per_layer=100, num_layers=3, num_timesteps=15)
        
        # Train it 
        model.train(session, max_epochs=5, max_steps=500, print_logs=True)
    
        
        # Let it generate a text
        generated = model.sample(session, "The ", num_steps=100)
        print("The " + generated)

This will output:

    Epoch: 1, Step: 100, Avg. Train Loss: 4.477590560913086
    Epoch: 1, Step: 200, Avg. Train Loss: 3.0998141765594482
    Epoch: 1, Step: 300, Avg. Train Loss: 2.977341890335083
    Epoch: 1, Step: 400, Avg. Train Loss: 2.8836493492126465
    Epoch: 1, Step: 500, Avg. Train Loss: 2.775566816329956
    The e e e e e e e e e e e e e e e e e ee ee ee ee ee e e e e e e e e e e e e ee ee ee ee ee e e e e e e e
    
## Requirements

    tensorflow >= 1.0