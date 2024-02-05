import numpy as np
import matplotlib.pyplot as plt
import emcee
import acor._acor as acor


def rosenbrock_lnL(params):
    x1, x2 = params
    lnL = ((-100*pow((x2-pow(x1, 2)), 2)) + pow((1-x1), 2))/20
    return lnL


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
    def __init__(self, n_walkers, n, lnp_fn, args=None, a=1.5, temp=1., gaussian=False):
        self.n = n  # number of dimensions
        self.n_walkers = n_walkers  # number of walkers
        self.lnp_fn = lnp_fn  # log probability function
        self.a = a
        self.temp = temp
        self.gaussian = gaussian
        self.acc_rate = 0.
        self.accepted = 0
        self.rejected = 0
        self.chain = []
    def one_qmc_sweep(self, curr_states):
        #states, n_walkers, temp, acc_ratio, a, gaussian
        """
        performs a single quadratic monte carlo step, sweep over all n_walkers

        :param acc_ratio: the acceptance ratio
        :param states: state class
        :param n_walkers: int, number of walkers
        :param gaussian: boolean, gaussian sampling or not
        :param a: float, width of the distribution
        :param temp: float, equivalent of the temperature in a boltzmann distribution
        :return: the resulting state of performing one qmc step for all walkers
        """

        if self.n_walkers < 3 or self.n_walkers < (2 * self.n):
            raise ValueError("error: the number of walkers is too small".format(self.n_walkers))

        new_state = np.zeros((self.n_walkers, self.n))  # initialize a new state

        n_invalid = 0
        prev_state = curr_states[-1]

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
            new_state[i] = wi * prev_state[i] + wj * prev_state[j] + wk * prev_state[k]


            # TODO: mind this, the following line should check the validity of the new state generated
            lnp_new = self.lnp_fn(new_state[i])
            ok = False
            if not np.isfinite(lnp_new):
                n_invalid += 1
                if n_invalid > 100:
                    raise ValueError("parameters repeatedly invallid")
            else:
                n_invalid = 0
                ok = True

            dlnp = lnp_new - self.lnp_fn(prev_state[i])
            accept = False

            # the following block is equivalent to the acceptance function
            # start checking acceptance
            p = (dlnp / self.temp) + np.log(np.power(abs(wi), self.n))
            accept = p > np.log(np.random.uniform())
            # end checking acceptance

            if ok and accept:  # check the acceptance of a newly generated state
                # n_accepted += 1 TODO: mind this, should have something that checks the number of accepted states
                self.accepted += 1
            else:
                new_state[i] = prev_state[i]
                self.rejected += 1
                #curr_states[i].y = new_state.y
        curr_states.append(new_state)

    def run_qmc(self, starting_guesses, n_steps):
        states = starting_guesses
        self.chain.append(starting_guesses)
        if self.n_walkers < 3:
            raise ValueError("n_walkers too small: {}".format(self.n_walkers))

        for i_step in range(n_steps):
            self.one_qmc_sweep(self.chain)
        self.chain = np.array(self.chain)[1:]


emcee_trace = []

for i in range(1):
    # Here we'll set up the computation. emcee combines multiple "walkers",
    # each of which is its own MCMC chain. The number of trace results will
    # be nwalkers * nsteps

    ndim = 2  # number of parameters in the model
    nwalkers = 5  # number of MCMC walkers
    nburn = 0  # "burn-in" period to let chains stabilize
    nsteps = 100000  # number of MCMC steps to take

    # set theta near the maximum likelihood, with
    starting_guesses = np.random.random((nwalkers, ndim))

    # Here's the function call where all the work happens:
    # we'll time it using IPython's %time magic

    sampler = emcee.EnsembleSampler(nwalkers, ndim, rosenbrock_lnL)
    sampler.run_mcmc(starting_guesses, nsteps)
    #emcee_trace.append(sampler.chain[:, nburn:, 0].T)
    emcee_trace.append(sampler.chain[:, nburn:, :].reshape(-1, ndim))


def next_pow_two(n):
    i = 1
    while i < n:
        i = i << 1
    return i


def autocorr_func_1d(x, norm=True):
    x = np.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")
    n = next_pow_two(len(x))

    # Compute the FFT and then (from that) the auto-correlation function
    f = np.fft.fft(x - np.mean(x), n=2 * n)
    acf = np.fft.ifft(f * np.conjugate(f))[: len(x)].real
    acf /= 4 * n

    # Optionally normalize
    if norm:
        acf /= acf[0]

    return acf


def auto_window(taus, c):
    m = np.arange(len(taus)) < c * taus
    if np.any(m):
        return np.argmin(m)
    return len(taus) - 1


def autocorr_gw2010(y, c=5.0):
    f = autocorr_func_1d(np.mean(y, axis=0))
    taus = 2.0 * np.cumsum(f) - 1.0
    window = auto_window(taus, c)
    return taus[window]


emcee_trace = np.array(emcee_trace)
emcee_chain = emcee_trace[0].T
print(np.shape(emcee_chain))
emcee_tau_x1, _, _ = acor.acor(emcee_chain[0], 10)  # emcee.autocorr.integrated_time(emcee_chain)
print(emcee_tau_x1)
emcee_tau_x2, _, _ = acor.acor(emcee_chain[1], 10)  # emcee.autocorr.integrated_time(emcee_chain)
print(emcee_tau_x2)

qmc_trace = []

for i in range(1):
    # Here we'll set up the computation. emcee combines multiple "walkers",
    # each of which is its own MCMC chain. The number of trace results will
    # be nwalkers * nsteps

    ndim = 2  # number of parameters in the model
    nwalkers = 5  # number of MCMC walkers
    nburn = 0  # "burn-in" period to let chains stabilize
    nsteps = 100000  # number of MCMC steps to take

    # set theta near the maximum likelihood, with
    starting_guesses = np.random.random((nwalkers, ndim))

    # Here's the function call where all the work happens:
    # we'll time it using IPython's %time magic

    sampler = QMCSampler(nwalkers, ndim, rosenbrock_lnL)
    sampler.run_qmc(starting_guesses, nsteps)

    qmc_trace.append(sampler.chain[nburn:, :, :].reshape(-1, ndim).T)

qmc_trace = np.array(qmc_trace)
qmc_chain = qmc_trace[0]
print(np.shape(qmc_chain))
qmc_tau_x1,_,_ = acor.acor(qmc_chain[0], 10)
qmc_tau_x2,_,_ = acor.acor(qmc_chain[1], 10)
print(qmc_tau_x1)
print(qmc_tau_x2)
