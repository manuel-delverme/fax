from typing import Callable

import jax.experimental.optimizers
from jax import numpy as np
from jax import tree_util, lax

from fax import utils
from fax.utils import LagrangianParameters


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
def adam_extragradient_optimizer(step_sizes, betas=(0.5, 0.99), weight_norm=0.0, grad_clip=False, eps=1e-8, use_adam=True) -> (Callable, Callable, Callable):
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

    step_size_p, step_size_x, step_size_y = step_sizes
    step_size_p = jax.experimental.optimizers.make_schedule(step_size_p)
    step_size_x = jax.experimental.optimizers.make_schedule(step_size_x)
    step_size_y = jax.experimental.optimizers.make_schedule(step_size_y)

    def init(init_values):
        if use_adam:
            exp_avg = utils.tree_zero_like(init_values)
            exp_avg_sq = utils.tree_zero_like(init_values)
            return init_values, (exp_avg, exp_avg_sq)
        else:
            return init_values, None

    def update(step, grad_fns, state, batch):
        (x0, y0), grad_state = state
        step_sizes = step_size_p(step), step_size_x(step), step_size_y(step)

        grad_state, xbar, ybar = half_step(batch, grad_fns, grad_state, step, step_sizes, x0, x0, y0, y0)
        grad_state, x1, y1 = half_step(batch, grad_fns, grad_state, step, step_sizes, x0, xbar, y0, ybar)
        return (x1, y1), grad_state

    def half_step(batch, grad_fns, grad_state, step, step_sizes, x0, xbar, y0, ybar):
        if use_adam:
            (delta_x, delta_y), grad_state = adam_step(betas, eps, step_sizes, grad_fns, grad_state, xbar, ybar, step, weight_norm, grad_clip, batch)
        else:
            (delta_x, delta_y) = sgd_step(step_sizes, grad_fns, xbar, ybar, weight_norm, grad_clip, batch)
        del xbar
        del ybar
        x1 = sub(x0, delta_x)
        y1 = add(y0, delta_y)
        del delta_x
        del delta_y
        return grad_state, x1, y1

    def get_params(state):
        x, _opt_state = state
        return LagrangianParameters(*x)

    return init, update, get_params


def adam_step(betas, eps, step_sizes, grads_fn, grad_state, x, y, step, weight_norm, grad_clip, batch):
    exp_avg, exp_avg_sq = grad_state
    beta1, beta2 = betas
    grads = grads_fn(utils.LagrangianParameters(x, y), batch)

    if grad_clip:
        grads = jax.experimental.optimizers.clip_grads(grads, grad_clip)
    if weight_norm:
        gx, gy = grads
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

    delta_p = multiply_constant(step_sizes[0])(step_improvement.constr_params.theta)
    delta_x = multiply_constant(step_sizes[1])(step_improvement.constr_params.x)
    delta_y = multiply_constant(step_sizes[2])(step_improvement.multipliers)

    delta = utils.LagrangianParameters(utils.ConstrainedParameters(delta_p, delta_x), delta_y)

    grad_state = exp_avg, exp_avg_sq
    return delta, grad_state


def sgd_step(step_sizes, grads_fn, x, y, weight_norm, grad_clip, batch):
    grads = grads_fn(utils.LagrangianParameters(x, y), batch)
    if grad_clip:
        grads = jax.experimental.optimizers.clip_grads(grads, grad_clip)
    if weight_norm:
        gx, gy = grads
        grads = (gx + multiply_constant(weight_norm)(x), gy)
    delta0 = multiply_constant(step_sizes[0])(grads[0][0]), multiply_constant(step_sizes[1])(grads[0][1])
    delta0 = grads[0].__class__(*delta0)
    delta = delta0, multiply_constant(step_sizes[2])(grads[1])
    delta = utils.LagrangianParameters(*delta)
    return delta
