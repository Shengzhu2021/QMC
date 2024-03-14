# this file defines a state for mcmc to optimize
import numpy as np


class QMCState:
    def __init__(self, n, lnp_fn, params, valid_fn=None):
        self.lnL = None
        self.n = n # dimension of the state
        self.p = params # the data/parameter this state contains
        self.lnp_fn = lnp_fn # the log_probability function
        self.valid_fn = valid_fn # the function that checks the validity of the state
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
        """
        evaluates the log probablity of this state
        """
        if self.valid():
            self.lnL = self.lnp_fn(self.p)
        else:
            self.lnL = None
            raise ValueError('the parameter is invalid')


def set_up_starting_guesses(state, guess):
    """
    set up the qmc states version of the starting guesses given the starting guesses of the parameter
    guesses should have shape (nwalkers, ndim)
    """
    shape = np.shape(guess)
    starting_guesses = []
    row = shape[0]
    for r in range(row):
        state_rc = QMCState(state.n_dim(), state.lnp_fn, guess[r])
        starting_guesses.append(state_rc)
    return np.array(starting_guesses)

def states2array(states):
    r, c = np.shape(states)
    arr = np.zeros_like(states)
    arr = []
    for i in range(r):
        for j in range(c):
            params = states[i, j].p
            arr.append(params)
    return np.array(arr)