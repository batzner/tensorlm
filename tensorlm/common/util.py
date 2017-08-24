import tensorflow as tf

def get_chunks(items, size):
    """Get successive chunks from items."""
    chunks = []
    for i in range(0, len(items), size):
        chunks.append(items[i:i + size])
    return chunks




def restore_possible(out_dir):
    ckpt = tf.train.get_checkpoint_state(out_dir)
    return ckpt and ckpt.model_checkpoint_path