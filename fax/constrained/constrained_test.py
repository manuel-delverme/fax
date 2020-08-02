import jax
import jax.experimental.optimizers
import jax.nn
import jax.scipy.special
import jax.test_util
import jax.tree_util
from absl.testing import absltest

import fax
import fax.test_util
from fax.competitive import extragradient

jax.config.update("jax_enable_x64", True)
test_params = dict(rtol=1e-4, atol=1e-4, check_dtypes=False)
convergence_params = dict(rtol=1e-7, atol=1e-7)
benchmark = list(fax.test_util.load_HockSchittkowski_models())


def eg_solve(lagrangian, convergence_test, get_x, initial_values, max_iter=100000000, metrics=(), lr=None):
    if lr is None:
        lr = jax.experimental.optimizers.inverse_time_decay(1e-1, 500, 0.3, staircase=True)

    optimizer_init, optimizer_update, optimizer_get_params = extragradient.adam_extragradient_optimizer(
        step_size=lr
        # step_size_y=jax.experimental.optimizers.inverse_time_decay(5e-2, 500, 0.1, staircase=True),
    )

    @jax.jit
    def update(i, opt_state):
        grad_fn = jax.grad(lagrangian, (0, 1))
        return optimizer_update(i, grad_fn, opt_state)

    # solution = fax.loop.fixed_point_iteration(
    solution = fax.loop._debug_fixed_point_iteration(
        init_x=optimizer_init(initial_values),
        func=update,
        convergence_test=convergence_test,
        max_iter=max_iter,
        get_params=optimizer_get_params,
        # metrics=metrics,
    )
    x, multipliers = get_x(solution)
    return x, multipliers


if __name__ == "__main__":
    absltest.main()
