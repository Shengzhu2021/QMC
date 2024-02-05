# this script does blocking analysis on a chain
import numpy as np


class Blocking(object):
    def __init__(self, timeseries, fn, max_block_size=None, n_blocks=None, block_gap=1):
        if max_block_size:
            self.mbs = max_block_size
        elif n_blocks:
            self.mbs = len(timeseries) // n_blocks
        else:
            self.mbs = len(timeseries) // 4
        self.fn = fn  # the function used to evaluate the given parameters, in example.py, this is the function that calculates potential energy from the parameters
        self.ts = np.array([fn(timeseries[i].p) for i in range(len(timeseries))])
        self.blocked_ts = None
        self.bg = block_gap  # differnece of each adjacent block size 

    def statistics(self):
        """
        calculates the relevant stats of the blocked timeseries
        """
        # initalizing the array for means and variance of each block size
        block_means = []
        block_vars = []
        blocks = np.linspace(1, self.mbs, self.mbs//self.bg, dtype=int)  # compute all block sizes needed
        for bs in blocks:
            n_blocks = len(self.ts) // bs # find the number of blocks
            # initalizing the array that holds the mean of each block
            mean_arr = []
            # for each block of size bs, compute the mean
            for i in range(1, n_blocks + 1):
                i_min = (i - 1) * bs
                i_max = i_min + bs
                mean_arr.append(np.mean(self.ts[i_min:i_max]))

            block_means.append(np.mean(mean_arr)) # mean of the means
            block_vars.append(np.sqrt(np.var(mean_arr)/(n_blocks - 1))) # standard error of the means
        return blocks, block_vars, block_means
