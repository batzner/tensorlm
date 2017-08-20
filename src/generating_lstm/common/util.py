import numpy as np


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def one_hot(hot_index, length):
    result = np.zeros(length)
    result[hot_index] = 1
    return result