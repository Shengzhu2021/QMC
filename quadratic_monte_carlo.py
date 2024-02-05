# this script contains the function to perform one quadratic monte carlo sweep
# this is for now a generic state of the function, will be altered to accommodate tge ensemble sampler class

import numpy as np
from state import QMCState


def l_interpolation(x, x0, x1, x2):
    """
    performs a lagrangian interpolation on 3 points and return the lagrangian weighting at x

    :param x: where the lagrangian weighting function is evaluated at
    :param x0: point 0
    :param x1: point 1
    :param x2: point 2
    :return: lagrangian weighting at x
    """
    return ((x - x1) / (x0 - x1)) * ((x - x2) / (x0 - x2))


class QMCSampler(object):
    def __init__(self, n_walkers, state, a=1.5, temp=1., gaussian=False):
        self.n = state.n  # number of dimensions
        self.n_walkers = n_walkers  # number of walkers
        self.valid_fn = state.valid_fn
        self.lnp_fn = state.lnp_fn  # log probability function
        self.a = a
        self.temp = temp
        self.gaussian = gaussian
        self.acc_rate = 0.
        self.accepted = 0
        self.rejected = 0
        self.chain = []

    def one_qmc_sweep(self, curr_states):
        """
        performs a single quadratic monte carlo step, sweep over all n_walkers
        """

        if self.n_walkers < 3 or self.n_walkers < (2 * self.n):
            raise ValueError("error: the number of walkers is too small".format(self.n_walkers))

        new_states = []  # initialize a new state

        n_invalid = 0
        prev_states = curr_states[-1]

        for i in range(self.n_walkers):  # iterate through each walker

            # choose 2 other walkers at random
            j = i
            k = i
            while j == i:
                j = np.random.randint(0, self.n_walkers)

            while k == j or k == i:
                k = np.random.randint(0, self.n_walkers)

            tj = -1.
            tk = 1.

            # determine ti and t_new, and introduce scale to the quadratic fit
            if self.gaussian:
                ti = np.random.normal(0, self.a)
                t_new = np.random.normal(0, self.a)
            else:
                ti = np.random.uniform(-self.a, self.a)
                t_new = np.random.uniform(-self.a, self.a)

            wi = l_interpolation(t_new, ti, tj, tk)
            wj = l_interpolation(t_new, tj, tk, ti)
            wk = l_interpolation(t_new, tk, ti, tj)

            # fully define the new state using lagrange interpolated parabola
            new_state = QMCState(self.n, self.lnp_fn,
                                     wi * prev_states[i].p + wj * prev_states[j].p + wk * prev_states[k].p)

            lnp_new = self.lnp_fn(new_state.p)
            ok = False
            if not np.isfinite(lnp_new):
                n_invalid += 1
                if n_invalid > 100:
                    raise ValueError("parameters repeatedly invalid")
            if self.valid_fn:
                if not self.valid_fn(new_state.p):
                    n_invalid += 1
                    if n_invalid > 100:
                        raise ValueError("parameters repeatedly invalid")
            else:
                n_invalid = 0
                ok = True

            dlnp = lnp_new - self.lnp_fn(prev_states[i].p)

            # the following block is equivalent to the acceptance function
            # start checking acceptance
            p = (dlnp / self.temp) + np.log(np.power(abs(wi), self.n))
            accept = p > np.log(np.random.uniform())
            # end checking acceptance

            if ok and accept:  # check the acceptance of a newly generated state
                # n_accepted += 1 TODO: mind this, should have something that checks the number of accepted states
                new_states.append(new_state)
                self.accepted += 1
            else:
                new_states.append(prev_states[i])
                self.rejected += 1
        curr_states.append(np.array(new_states))

    def run_qmc(self, starting_guesses, n_steps):
        self.chain.append(starting_guesses)
        if self.n_walkers < 3:
            raise ValueError("n_walkers too small: {}".format(self.n_walkers))

        for i_step in range(n_steps):
            self.one_qmc_sweep(self.chain)
        self.chain = np.array(self.chain)[1:]
