# this file defines a state for mcmc to optimize
import numpy as np


class QMCState:
    def __init__(self, n, lnp_fn, params, valid_fn=None):
        self.lnL = None
        self.n = n
        self.p = params
        self.lnp_fn = lnp_fn
        self.valid_fn = valid_fn
        self.evaluate()

    def n_dim(self):
        return self.n

    def valid(self):
        if self.valid_fn:
            valid = self.valid_fn()
            if isinstance(bool(valid), bool):
                return valid
            else:
                raise ValueError('invalid return from valid function')
        else:
            return True

    def __getitem__(self, i):
        return self.p[i]

    def __setitem__(self, i, value):
        self.p[i] = value
        self.evaluate()

    def __call__(self):
        if self.n < 2:
            return 0.0
        self.evaluate()
        return self.lnL

    def __repr__(self):
        return repr(self.p)

    def evaluate(self):
        if self.valid():
            self.lnL = self.lnp_fn(self.p)
        else:
            self.lnL = None
            raise ValueError('the parameter is invalid')


def set_up_starting_guesses(state, guess):
    shape = np.shape(guess)
    starting_guesses = []
    row = shape[0]
    for r in range(row):
        state_rc = QMCState(state.n_dim(), state.lnp_fn, guess[r])
        starting_guesses.append(state_rc)
    return np.array(starting_guesses)
