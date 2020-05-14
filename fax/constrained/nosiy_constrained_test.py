import jax.tree_util
from absl.testing import absltest

import fax.constrained.constrained_test
import fax.test_util

jax.config.update("jax_enable_x64", True)
test_params = dict(rtol=1e-4, atol=1e-4, check_dtypes=False)
convergence_params = dict(rtol=1e-5, atol=1e-5)


class NoisyEGTest(fax.constrained.constrained_test.EGTest):
    pass


if __name__ == "__main__":
    absltest.main()
