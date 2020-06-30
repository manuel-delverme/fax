def _debug_fixed_point_iteration(init_x, func, convergence_test, max_iter, batched_iter_size=1,
                                 unroll=False, f=None, get_params=lambda x: x) -> FixedPointSolution:
    xs = []
    ys = []
    js = []

    def while_loop(cond_fun, body_fun, init_vals):
        loop_state = init_vals

        iterations, (x_new, _optimizer_state), prev_sol = loop_state
        player_x_new, player_y_new = x_new

        xs.append(player_x_new)
        ys.append(player_y_new)
        if f is not None:
            js.append(f(*x_new))

        while True:
            loop_state = body_fun(loop_state)
            iterations, (x_new, _optimizer_state), prev_sol = loop_state
            if iterations % 50 == 0 and iterations < 1000 or (iterations % 200 == 0):
                plot_process(js, xs, ys)
            player_x_new, player_y_new = x_new

            xs.append(player_x_new)
            ys.append(player_y_new)
            if f is not None:
                js.append(f(*x_new))

            if not cond_fun(loop_state):
                return loop_state

    jax_while_loop = jax.lax.while_loop
    jax.lax.while_loop = while_loop

    solution = fixed_point_iteration(init_x, func, convergence_test, max_iter, batched_iter_size, unroll, get_params)

    jax.lax.while_loop = jax_while_loop

    plot_process(js, xs, ys)
    return solution


def plot_process(js, xs, ys):
    import matplotlib.pyplot as plt
    plt.grid(True)
    xs = np.array(xs)
    ts = np.arange(len(xs))
    plt.title("xs")
    plt.plot(ts, xs)
    plt.scatter(np.zeros_like(xs), xs)
    plt.show()
    # plt.title("ys")
    # plt.plot(ts, ys)
    # plt.show()
    # if js:
    #     plt.title("js")
    #     plt.plot(ts, js)
    # plt.show()
