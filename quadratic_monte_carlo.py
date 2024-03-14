# this script contains the function to perform one quadratic monte carlo sweep
# this is for now a generic state of the function, will be altered to accommodate tge ensemble sampler class

import numpy as np
from state import QMCState
import os


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
    def __init__(self, n_walkers, state, a=1.5, temp=1., gaussian=False, toFile=False, filename=None, filepath=None):
        self.n = state.n  # number of dimensions
        self.n_walkers = n_walkers  # number of walkers
        self.valid_fn = state.valid_fn
        self.lnp_fn = state.lnp_fn  # log probability function
        self.a = a
        self.temp = temp # optional temperature argument
        self.gaussian = gaussian # whether to use gaussian random or not
        self.toFile = toFile
        self.filename = filename
        self.filepath = filepath
        self.acc_rate = 0.
        self.accepted = 0
        self.rejected = 0
        self.chain = []

        if toFile:
            if not filepath:
                self.filepath = os.getcwd()
            if not filename:
                self.filename = 'qmc_chain'
            f = open(self.filepath + '\\' + self.filename + '.txt', "w")   # 'r' for reading and 'w' for writing
            f.write("\n" + f.name)    # Write inside file 
            f.close()  

    def get_chain(self):
        if not self.toFile:
            return self.chain
        f = open(self.filepath + '/' + self.filename + '.txt', "r")
        chain = np.loadtxt(f, delimiter=',')
        f.close()
        return chain

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
            # do lagrangian interpolation
            wi = l_interpolation(t_new, ti, tj, tk)
            wj = l_interpolation(t_new, tj, tk, ti)
            wk = l_interpolation(t_new, tk, ti, tj)

            # fully define the new state using lagrange interpolated parabola
            new_state = QMCState(self.n, self.lnp_fn,
                                     wi * prev_states[i].p + wj * prev_states[j].p + wk * prev_states[k].p)

            # check validty of the new state
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
            p = (dlnp / self.temp) + np.log(np.power(abs(wi), self.n)) # the acceptance ratio
            accept = p > np.log(np.random.uniform())
            # end checking acceptance

            if ok and accept:  # check the acceptance of a newly generated state
                self.accepted += 1
                new_states.append(new_state)
            else:
                new_states.append(prev_states[i])
                self.rejected += 1
        curr_states.append(np.array(new_states))

    def one_qmc_sweep_toFile(self, prev_states):
            """
            one qmc sweep toFile version
            """
            if self.n_walkers < 3 or self.n_walkers < (2 * self.n):
                raise ValueError("error: the number of walkers is too small".format(self.n_walkers))

            new_states = []  # initialize a new state

            n_invalid = 0 

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
                # do lagrangian interpolation
                wi = l_interpolation(t_new, ti, tj, tk)
                wj = l_interpolation(t_new, tj, tk, ti)
                wk = l_interpolation(t_new, tk, ti, tj)

                # fully define the new state using lagrange interpolated parabola
                new_state = wi * prev_states[i] + wj * prev_states[j] + wk * prev_states[k]

                # check validty of the new state
                lnp_new = self.lnp_fn(new_state)
                ok = False
                if not np.isfinite(lnp_new):
                    n_invalid += 1
                    if n_invalid > 100:
                        raise ValueError("parameters repeatedly invalid")
                if self.valid_fn:
                    if not self.valid_fn(new_state):
                        n_invalid += 1
                        if n_invalid > 100:
                            raise ValueError("parameters repeatedly invalid")
                else:
                    n_invalid = 0
                    ok = True

                dlnp = lnp_new - self.lnp_fn(prev_states[i])

                # the following block is equivalent to the acceptance function
                # start checking acceptance
                p = (dlnp / self.temp) + np.log(np.power(abs(wi), self.n)) # the acceptance ratio
                accept = p > np.log(np.random.uniform())
                # end checking acceptance

                if ok and accept:  # check the acceptance of a newly generated state
                    self.accepted += 1
                    new_states.append(new_state)
                else:
                    new_states.append(prev_states[i])
                    self.rejected += 1
            f = open(self.filepath + '\\' + self.filename + '.txt', "a")
            for n in range(self.n_walkers):
                f.write(', '.join(list(np.array(new_states[n], dtype=str))) + '\n')
            f.close()
            return new_states

    def run_qmc(self, starting_guesses, n_steps):
        """
        runs the qmc for n_steps
        """
        if not self.toFile:
            self.chain.append(starting_guesses)
            if self.n_walkers < 3:
                raise ValueError("n_walkers too small: {}".format(self.n_walkers))

            for i_step in range(n_steps):
                self.one_qmc_sweep(self.chain)
            self.chain = np.array(self.chain)[1:]
            self.acc_rate = self.accepted/(self.accepted+self.rejected)
        else:
            prev_states = starting_guesses
            starting_guesses = np.array(starting_guesses, dtype=str)
            f = open(self.filepath + '/' + self.filename + '.txt', "w")
            for n in range(self.n_walkers):
                f.write(', '.join(list(starting_guesses[n])) + '\n')
            f.close()
            for _ in range(n_steps):
                prev_states = self.one_qmc_sweep_toFile(prev_states)
            
        
