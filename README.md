# fax: fixed-point jax 

Implicit and [competitive differentiation](https://optrl2019.github.io/assets/accepted_papers/70.pdf) in [JAX](https://github.com/google/jax).

Our "competitive differentiation" approach uses [Competitive Gradient Descent](https://arxiv.org/abs/1905.12103) to solve the equality-constrained nonlinear program associated with the fixed-point problem. A standalone implementation of CGD is provided under [fax/competitive/cga.py](fax/competitive/cga.py) and the equality-constrained solver derived from it can be accessed via `fax.constrained.cga_lagrange_min` or `fax.constrained.cga_ecp`. An implementation of implicit differentiation based on [Christianson's](https://doi.org/10.1080/10556789408805572) two-phases reverse accumulation algorithm can also be obtained with the function `fax.implicit.two_phase_solver`.

See [fax/constrained/constrained_test.py](fax/constrained/constrained_test.py) for examples. Please note that the API is subject to change.
