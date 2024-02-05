from quadratic_monte_carlo import QMCSampler
from state import QMCState, set_up_starting_guesses
import numpy as np
import matplotlib.pyplot as plt

R = 1.
m = 6.
ndim = 6
m2 = 2 * m


def ring_lnp(params):
    rho = np.sqrt(params[0] ** 2 + params[1] ** 2)
    v = pow(m2 * (rho - R), m2)
    i = 2
    while i < ndim:
        v += pow(m2 * params[i], m2)
        i += 1
    v -= 0.01 * params[0]
    return -v / 0.01


A = np.array([R, 0, 0, 0, 0, 0])
B = np.array([-R, 0, 0, 0, 0, 0])
As = []
Bs = []

for i in range(20):
    As.append(A)
    Bs.append(B)
As = np.array(As)
Bs = np.array(Bs)


def rosenbrock_lnL(params):
    x1, x2 = params
    lnL = ((-100 * pow((x2 - pow(x1, 2)), 2)) + pow((1 - x1), 2)) / 20
    return lnL


def harmonic_lnp(params):
    n = len(params)
    v = 0
    for j in range(n):
        v += params[j] * params[j]
    return -(v / 0.01)


qmc_trace_harmonic = []
harmonic_state = QMCState(6, harmonic_lnp, np.zeros(6))

for i in range(1):
    # Here we'll set up the computation. emcee combines multiple "walkers",
    # each of which is its own MCMC chain. The number of trace results will
    # be nwalkers * nsteps

    ndim = 6  # number of parameters in the model
    nwalkers = 20  # number of MCMC walkers
    nburn = 5000  # "burn-in" period to let chains stabilize
    nsteps = 10000  # number of MCMC steps to take

    # set theta near the maximum likelihood, with
    sg = np.random.random((nwalkers, ndim))
    starting_guesses = set_up_starting_guesses(harmonic_state, sg)

    # Here's the function call where all the work happens:
    # we'll time it using IPython's %time magic

    sampler = QMCSampler(nwalkers, harmonic_state)
    sampler.run_qmc(starting_guesses, nsteps)

    qmc_trace_harmonic.append(sampler.chain[nburn:, :])

qmc_trace_harmonic = np.array(qmc_trace_harmonic)

v_arr = []
for k in range(1):
    for i in range(5000):
        for j in range(20):
            v = -qmc_trace_harmonic[k, i, j].lnL*0.01
            v_arr.append(v)
avg_v = np.mean(v_arr)
print(avg_v)
print(np.std(v_arr))
