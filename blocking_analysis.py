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
        self.ts = np.array([fn(timeseries[i].p) for i in range(len(timeseries))])
        self.fn = fn
        self.blocked_ts = None
        self.bg = block_gap

    def statistics(self):
        """
        calculates the relevant stats of the blocked timeseries
        """
        block_means = []
        block_vars = []
        blocks = np.linspace(1, self.mbs, self.mbs//self.bg, dtype=int)
        print(blocks)
        for bs in blocks:
            n_blocks = len(self.ts) // bs
            mean_arr = []

            for i in range(1, n_blocks + 1):
                i_min = (i - 1) * bs
                i_max = i_min + bs
                mean_arr.append(np.mean(self.ts[i_min:i_max]))

            block_means.append(np.mean(mean_arr))
            block_vars.append(np.var(mean_arr)/(n_blocks - 1))
        return blocks, block_vars, block_means
