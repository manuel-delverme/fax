import copy

import jax.numpy as jnp
import jax.test_util
import numpy as onp
from absl.testing import absltest
from jax import random
from jax.config import config
from jax.experimental.stax import softmax
from jax.random import bernoulli
from jax.scipy.special import logsumexp

from fax.constrained import slsqp_ecp

config.update("jax_enable_x64", True)
config.update('jax_disable_jit', True)

ojnp_log = jnp.log
jnp.log = lambda x: ojnp_log(1e-8 + x)


# finding reward function
class CGATest(jax.test_util.JaxTestCase):

    def __init__(self, arg):
        super().__init__(arg)
        self.key = random.PRNGKey(0)  # PRNG key for sampling trajectories

    # @parameterized.parameters(
    #     {'method': implicit_ecp,
    #      'kwargs': {'max_iter': 1000, 'lr_func': 0.01, 'optimizer': optimizers.adam}},
    #     {'method': cga_ecp, 'kwargs': {'max_iter': 1000, 'lr_func': 0.01, 'lr_multipliers': 0.925}},
    #     {'method': slsqp_ecp, 'kwargs': {'max_iter': 1000}},
    # )

    def test_omd(self):
        true_transition = jnp.array([[[0.7, 0.3], [0.2, 0.8]],
                                     [[0.99, 0.01], [0.99, 0.01]]])
        true_reward = jnp.array(([[-0.45, -0.1],
                                  [0.5, 0.5]]))
        temperature = 1e-2
        true_discount = 0.9
        traj_len = 10
        initial_distribution = jnp.ones(2) / 2
        policy_expert = jnp.array(([[0.4, 0.6],
                                    [0.6, 0.4]]))

        # # binomial for jax
        # # @jax.jit
        # def bernoulli(p=0.5):
        #     key, subkey = random.split(self.key)
        #     self.key = subkey
        #     u = random.uniform(self.key)
        #     return jnp.where(u <= p, 0, 1).astype(int)

        def get_new_key():
            key, subkey = random.split(self.key)
            self.key = subkey

        # trajectory rollout
        # @jax.jit
        def roll_out(last_state, last_action, p, pi):
            get_new_key()
            s = bernoulli(self.key, p=p[last_action][last_state][0]).astype(int)
            get_new_key()
            a = bernoulli(self.key, p=pi[s][0]).astype(int)
            return (s, a)

        # @jax.jit
        def sample_trajectory(policy):
            get_new_key()
            s = bernoulli(self.key, p=initial_distribution[0]).astype(int)
            get_new_key()
            a = bernoulli(self.key, p=policy[s][0]).astype(int)
            traj = []
            traj.append((s, a))
            for i in range(traj_len - 1):
                s, a = roll_out(s, a, true_transition, policy)
                traj.append((s, a))
            return jnp.array(copy.deepcopy(traj))

        def smooth_bellman_optimality_operator(x, params):
            transition, reward, discount, temperature = params
            return reward + discount * jnp.einsum('ast,t->sa', transition, temperature *
                                                  logsumexp((1. / temperature) * x, axis=1))

        # @jax.jit
        # def generator(discriminator, data_pi):
        #     states = data_pi[:,0]
        #     actions = data_pi[:, 1]
        #     tmp = discriminator[states][jnp.arange(len(states)), actions]
        #     loss = jnp.log(1-tmp).sum()
        #     return -loss.astype(float) / float(traj_len)

        @jax.jit
        def generator(discriminator, data_pi):
            loss = 0
            for i in range(traj_len):
                s_pi, a_pi = data_pi[i]
                loss += jnp.log(1 - softmax(discriminator, axis=1)[s_pi][a_pi])
            return - loss / traj_len

        # # using lax.fori_loop
        # @jax.jit
        # def generator(discriminator, data_pi):
        #     def loop_body(i, val):
        #         loss = val
        #         s_pi, a_pi = data_pi[i]
        #         loss = loss + jnp.log(1 - discriminator[s_pi][a_pi])
        #         return loss
        #
        #     loss = lax.fori_loop(0, traj_len, loop_body, (0.0))
        #     return - loss / traj_len

        # @jax.jit
        def objective(x, params):

            # del params

            q = x[:2]
            discriminator_logits = x[2:]

            policy = softmax((1. / temperature) * q)  # [2, 2]
            discriminator = softmax((1. / temperature) * discriminator_logits)

            data_pi = sample_trajectory(policy)
            loss = generator(discriminator, data_pi)
            # print("policy: ", policy)
            # print("loss", loss)

            # gradient = jax.grad(generator)(discriminator, data_pi)

            # print(" discriminator gradeint: ", gradient.flatten())
            return loss

        # using lax.fori_loop
        # @jax.jit
        # def ratio_loss(discriminator, data_pi, data_expert):
        #     def body_fun(i,x):
        #         s_expert, a_expert = data_expert[i]
        #         s_pi, a_pi = data_pi[i]
        #         tmp = -jnp.log(discriminator[s_expert][a_expert]) - jnp.log(1 - discriminator[s_pi][a_pi])
        #         return x + tmp
        #
        #     loss = lax.fori_loop(0, traj_len, body_fun, (0.0))
        #     return loss / traj_len
        #

        # @jax.jit
        # def ratio_loss(discriminator, data_pi, data_expert):
        #     loss = 0
        #     for i in range(traj_len):
        #         s_expert, a_expert = data_expert[i]
        #         s_pi, a_pi = data_pi[i]
        #         loss += -jnp.log(discriminator[s_expert][a_expert]) - jnp.log(1 - discriminator[s_pi][a_pi])
        #     return loss/traj_len

        # @jax.jit
        def ratio_loss(discriminator, data_pi, data_expert):
            states_exp = data_expert[:, 0]
            actions_exp = data_expert[:, 1]
            states_pi = data_pi[:, 0]
            actions_pi = data_pi[:, 1]

            tmp_pi = discriminator[states_pi][jnp.arange(len(states_pi)), actions_pi]
            tmp_exp = discriminator[states_exp][jnp.arange(len(states_exp)), actions_exp]
            # loss = jnp.log(tmp_pi).sum() + jnp.log(1 - tmp_exp).sum()
            loss = jnp.linalg.norm(tmp_pi - tmp_exp, 2)
            return loss.astype(float) / float(traj_len)

        # @jax.jit
        def equality_constraints(x, params):
            # print("sum_x: ", x.sum())
            # print("sum_params: ", params.sum())

            q = x[:2]
            discriminator_logits = x[2:]
            reward_logits = params

            reward_hat = softmax((1. / temperature) * reward_logits)
            discriminator = softmax((1. / temperature) * discriminator_logits)
            policy = softmax((1. / temperature) * q)

            # def cons1(value):
            #     return smooth_bellman_optimality_operator(value, params) - value

            # constraint 1
            params = (true_transition, reward_hat, true_discount, temperature)
            constraint1 = smooth_bellman_optimality_operator(q, params) - q
            constraint1 = constraint1.flatten()

            # constraint 2
            data_pi = sample_trajectory(policy)
            data_expert = sample_trajectory(policy_expert)
            # loss_grad = jax.grad(ratio_loss)
            # constraint2 = loss_grad(discriminator, data_pi, data_expert)
            constraint2 = ratio_loss(discriminator, data_pi, data_expert)
            constraint2 = constraint2.flatten()

            # for debugging

            # print("constraint1 gradient: ", jax.jacfwd(cons1)(q))
            print("constraint1: ", constraint1.flatten())
            print("constraint2: ", constraint2.flatten())
            # print("q: ", q.flatten())
            # print("reward_logits:", reward_logits.flatten())
            # print("discriminator_logits:", discriminator_logits.flatten())
            # print("policy:", policy)
            # print("\n\n")
            return jnp.concatenate((constraint1, constraint2), axis=0)

        init_x = jnp.concatenate((onp.random.random(2), onp.random.random(2)), axis=0)
        # init_x = jnp.zeros((4, 2))

        initial_values = (
            jnp.array(onp.random.random((4, 2))),
            jnp.array(onp.random.random(true_reward.shape))  # reward parameters
        )

        # kwargs = {'max_iter': 1000, 'lr_func': 0.01, 'optimizer': optimizers.adam}
        # solution = implicit_ecp(objective, equality_constraints, initial_values, **kwargs)

        kwargs = {'max_iter': 100}
        solution = slsqp_ecp(objective, equality_constraints, initial_values, **kwargs)
        print(solution)


if __name__ == "__main__":
    absltest.main()
