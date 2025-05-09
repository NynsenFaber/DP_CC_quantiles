"""
k-ary Tree Noise Mechanism for Differentially Private Continual Counting.

This module defines the KaryTreeNoise class, which constructs a k-ary tree
over a data stream of length `max_time` and adds discrete Laplace (two-sided
geometric) noise at each node to provide pure ε-differential privacy for
continual prefix sum releases.

Features:
- Automatically selects branching factor `k` to minimize worst-case variance
  if not provided.
- Supports computation of noise terms (`prefix_noise_terms`) and their sum
  (`prefix_noise`) for any time t.
- Provides an upper bound on the variance of the noise via `variance_bound`.

Usage:
    from k_ary_tree_noise import KaryTreeNoise
    tree = KaryTreeNoise(eps=1.0, max_time=1000)
    noise = tree.prefix_noise(t)
"""

import numpy as np
from math import ceil, log
from typing import Optional

class KaryTreeNoise:
    def __init__(self, eps: float, max_time: int, k: Optional[int] = None):
        self.eps = eps
        # if k not provided, pick k that minimizes worst-case variance bound
        if k is None:
            best_k = 2
            best_var = float('inf')
            for cand in range(2, max_time + 1):
                H_c = ceil(log(max_time, cand))
                scale_c = ((H_c + 1) / 2) / eps
                b_c = np.exp(-1 / scale_c)
                var_node_c = 2 * b_c / (1 - b_c) ** 2
                max_terms_c = H_c * (cand - 1) + 2
                var_bound_c = max_terms_c * var_node_c
                if var_bound_c < best_var:
                    best_var = var_bound_c
                    best_k = cand
            k = best_k
        self.k = k
        self.H = ceil(log(max_time, self.k))
        # true sensitivity is (H+1)/2 across levels 0..H
        self.scale = ((self.H + 1) / 2) / eps
        # parameters for discrete Laplace (two-sided geometric)
        self.b = np.exp(-1 / self.scale)
        self.p = 1 - self.b
        # levels 0..H
        self.noise = [dict() for _ in range(self.H + 1)]

    def _get_noise(self, level: int, idx: int) -> float:
        if idx not in self.noise[level]:
            # sample discrete Laplace via difference of two Geom(p) draws
            g1 = np.random.geometric(self.p)
            g2 = np.random.geometric(self.p)
            self.noise[level][idx] = g1 - g2
        return self.noise[level][idx]

    def prefix_noise_terms(self, t: int):
        if not (1 <= t <= self.k**self.H):
            raise ValueError(f"t must be in [1, {self.k**self.H}]")

        # always include root
        yield self._get_noise(0, 0)

        rem = t - 1
        for level in range(1, self.H + 1):
            block_size = self.k ** (self.H - level)
            i = rem // block_size
            rem %= block_size

            # include the k-1 sibling noises
            for j in range(self.k):
                if j == i:
                    continue
                sign = +1 if j < i else -1
                yield sign * self._get_noise(level, j)

            # **if we’ve exactly hit the end of a block,**
            # add the noise for that full subtree and stop
            if rem == 0:
                yield self._get_noise(level, i)
                break

    def prefix_noise(self, t: int) -> float:
        return sum(self.prefix_noise_terms(t))

    def variance_bound(self) -> float:
        """
        Computes an upper bound on the variance of the noise added to any prefix sum output.
        """
        # variance of one discrete Laplace node: Var = 2b / (1 - b)^2
        var_node = 2 * self.b / (1 - self.b) ** 2
        # maximum number of noise terms: root + H*(k-1) siblings + final subtree noise
        max_terms = self.H * (self.k - 1) + 2
        return max_terms * var_node


# Example
if __name__ == "__main__":
    MAX_TIME = 23
    tree = KaryTreeNoise(eps=1.0, max_time=MAX_TIME)
    for t in range(1,MAX_TIME+1):
        print(f"t={t:3d}, noise={tree.prefix_noise(t)}")
