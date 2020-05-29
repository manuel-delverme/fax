import jax.numpy as np


# def relu(x):
#     return np.maximum(x, 0)


def leaky_relu(x):
    return np.maximum(x, 0.001 * x)
