from typing import Callable

import jax.experimental.optimizers
from jax import numpy as np
from jax import tree_util, lax


def division_constant(constant):
    def divide(a):
        return tree_util.tree_multimap(lambda _a: _a / constant, a)

    return divide


def multiply_constant(constant):
    def multiply(a):
        return tree_util.tree_multimap(lambda _a: _a * constant, a)

    return multiply


division = lambda _a, _b: tree_util.tree_multimap(lambda _a, _b: _a / _b, _a, _b)
add = lambda _a, _b: tree_util.tree_multimap(lambda _a, _b: _a + _b, _a, _b)
sub = lambda _a, _b: tree_util.tree_multimap(lambda _a, _b: _a - _b, _a, _b)


def mul(_a, _b):
    return tree_util.tree_multimap(lax.mul, _a, _b)


# mul = lambda a, b: tree_util.tree_multimap(lax.mul(a, b), a, b)
square = lambda _a: tree_util.tree_map(np.square, _a)


def extragradient_optimizer(*args, **kwargs) -> (Callable, Callable, Callable):
    return rprop_extragradient_optimizer(*args, **kwargs, use_rprop=False)


def rprop_extragradient_optimizer(step_size_x, step_size_y, proj_x=lambda x: x, proj_y=lambda y: y, use_rprop=True) -> (Callable, Callable, Callable):
    """Provides an optimizer interface to the extra-gradient method

    We are trying to find a pair (x*, y*) such that:

    f(x*, y) ≤ f(x*, y*) ≤ f(x, y*), ∀ x ∈ X, y ∈ Y

    where X and Y are closed convex sets.

    Args:
        init_values:
        step_size_x: TODO
        step_size_y: TODO
        f: Saddle-point function
        convergence_test:  TODO
        max_iter:  TODO
        batched_iter_size:  TODO
        unroll:  TODO
        proj_x: Projection on the convex set X
        proj_y: Projection on the convex set Y
        eps: rms prop eps
        gamma: rms prop gamma

    """
    step_size_x = jax.experimental.optimizers.make_schedule(step_size_x)
    step_size_y = jax.experimental.optimizers.make_schedule(step_size_y)

    def init(init_values):
        x0, y0 = init_values
        assert len(x0.shape) == (len(y0.shape) == 1 or not y0.shape)
        if not y0.shape:
            y0 = y0.reshape(-1)
        return (x0, y0), np.ones(x0.shape[0] + y0.shape[0])

    def update(i, grads, state):
        (x0, y0), grad_state = state
        step_sizes = step_size_x(i), step_size_y(i)

        delta_x, delta_y, _ = sign_adaptive_step(step_sizes, grads, grad_state, x0, y0, i, use_rprop=use_rprop)

        xbar = proj_x(x0 - delta_x)
        ybar = proj_y(y0 + delta_y)

        delta_x, delta_y, _ = sign_adaptive_step(step_sizes, grads, grad_state, xbar, ybar, i, use_rprop=use_rprop)
        x1 = proj_x(x0 - delta_x)
        y1 = proj_y(y0 + delta_y)

        return (x1, y1), grad_state

    def get_params(state):
        x, _ = state
        return x

    return init, update, get_params


# cannot be used because it requires grad in signature (instead of grad_fn)
# @jax.experimental.optimizers.optimizer
def adam_extragradient_optimizer(step_size, betas=(0.5, 0.99), weight_norm=0.1, eps=1e-8) -> (Callable, Callable, Callable):
    """Provides an optimizer interface to the extra-gradient method

    We are trying to find a pair (x*, y*) such that:

    f(x*, y) ≤ f(x*, y*) ≤ f(x, y*), ∀ x ∈ X, y ∈ Y

    where X and Y are closed convex sets.

    Args:
        init_values:
        step_size_x (float): x learning rate,
        step_size_y: (float): y learning rate,
        f: Saddle-point function
        convergence_test:  TODO
        max_iter:  TODO
        batched_iter_size:  TODO
        unroll:  TODO

        betas (Tuple[float, float]): coefficients used for computing running averages of gradient and its square.
        eps (float, optional): term added to the denominator to improve numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        ams_grad (boolean, optional): whether to use the AMSGrad variant of this algorithm from the paper `On the Convergence of Adam and Beyond`_

    """

    step_size = jax.experimental.optimizers.make_schedule(step_size)

    def init(init_values):
        exp_avg = tree_util.tree_map(lambda x: np.zeros(x.shape, x.dtype), init_values)
        exp_avg_sq = tree_util.tree_map(lambda x: np.zeros(x.shape, x.dtype), init_values)

        return init_values, (exp_avg, exp_avg_sq)

    def update(step, grad_fns, state):
        (x0, y0), grad_state = state
        step_sizes = step_size(step)

        (delta_x, delta_y), grad_state = adam_step(betas, eps, step_sizes, grad_fns, grad_state, x0, y0, step, weight_norm)
        # xbar = x0 - delta_x
        xbar = sub(x0, delta_x)
        ybar = add(y0, delta_y)

        (delta_x, delta_y), grad_state = adam_step(betas, eps, step_sizes, grad_fns, grad_state, xbar, ybar, step, weight_norm)
        # x1 = x0 - delta_x
        x1 = sub(x0, delta_x)
        y1 = add(y0, delta_y)

        return (x1, y1), grad_state

    def get_params(state):
        x, _opt_state = state
        return x

    return init, update, get_params


def sign_adaptive_step(step_size, grads_fn, grad_state, x, y, i, use_rprop=True):
    step_size_x, step_size_y = step_size

    grad_x0, grad_y0 = grads_fn(x, y)
    # the next part is to avoid ifs
    #  d |  d + 1 |  d - 1
    #  1 |    2   |    0
    # -1 |    0   |   -2
    if use_rprop:
        eta_plus = 1.2
        eta_minus = 0.5
        grads = np.concatenate((grad_x0, grad_y0))
        direction = np.sign(grad_state * grads)
        step_improvement_rate = (direction + 1) * eta_plus / 2. + (1 - direction) * eta_minus / 2
        eta_x = step_size_x * step_improvement_rate[:grad_x0.shape[0]]
        eta_y = step_size_y * step_improvement_rate[grad_x0.shape[0]:]
        grad_state = grads
    else:
        grad_state = None
        eta_x = step_size_x
        eta_y = step_size_y

    delta_x = eta_x * grad_x0
    delta_y = eta_y * grad_y0
    return delta_x, delta_y, grad_state


def adam_step(betas, eps, step_sizes, grads_fn, grad_state, x, y, step, weight_norm):
    exp_avg, exp_avg_sq = grad_state
    beta1, beta2 = betas
    (gx, gy) = grads_fn(x, y)
    grads = (gx + multiply_constant(weight_norm)(x), gy)

    bias_correction1 = 1 - beta1 ** (step + 1)
    bias_correction2 = 1 - beta2 ** (step + 1)

    def make_exp_smoothing(beta):
        def exp_smoothing(state, var):
            return state * beta + (1 - beta) * var

        return exp_smoothing

    exp_avg = tree_util.tree_multimap(make_exp_smoothing(beta1), exp_avg, grads)
    exp_avg_sq = tree_util.tree_multimap(make_exp_smoothing(beta2), exp_avg_sq, square(grads))

    corrected_moment = division_constant(bias_correction1)(exp_avg)
    corrected_second_moment = division_constant(bias_correction2)(exp_avg_sq)

    denom = tree_util.tree_multimap(lambda _var: np.sqrt(_var) + eps, corrected_second_moment)
    step_improvement = division(corrected_moment, denom)
    delta = multiply_constant(step_sizes)(step_improvement)

    grad_state = exp_avg, exp_avg_sq
    return delta, grad_state

