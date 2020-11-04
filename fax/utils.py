import collections

import jax.tree_util
from jax import tree_util, numpy as np


# def relu(x):
#     return np.maximum(x, 0)


def leaky_relu(x):
    return np.maximum(x, 0.001 * x)


class Batch(collections.namedtuple("Batch", "x y indices")):
    pass


class ConstrainedParameters(collections.namedtuple("ConstrainedParameters", "theta x")):
    def __sub__(self, other):
        return jax.tree_util.tree_multimap(lambda _a, _b: _a - _b, self, other)

    def __add__(self, other):
        return jax.tree_util.tree_multimap(lambda _a, _b: _a + _b, self, other)


class LagrangianParameters(collections.namedtuple("LagrangianParameters", "constr_params multipliers")):
    pass


def tree_zero_like(a):
    return tree_util.tree_map(lambda x: np.zeros(x.shape, x.dtype), a)
