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
        if max_time <= 1:
            raise ValueError(f"max_time must be at least 2")
        # if k not provided, pick k that minimizes worst-case variance bound
        if k is None:
            best_k = 2
            best_var = float('inf')
            for cand in range(2, max_time + 1):
                H_c = ceil(log(max_time, cand))
                scale_c = ((H_c + 1) / 2) / eps
                b_c = np.exp(-1 / scale_c)
                var_node_c = 2 * b_c / (1 - b_c) ** 2
                # improved variance bound: count only valid siblings per level
                max_terms_c = 2
                for level_c in range(1, H_c + 1):
                    block_size_c = cand ** (H_c - level_c)
                    valid_count = ceil(max_time / block_size_c)
                    # siblings per level capped by k-1
                    max_terms_c += (min(cand, valid_count) - 1)
                var_bound_c = max_terms_c * var_node_c
                # early stop: if variance increases significantly past the current best, break
                if var_bound_c > 10 * best_var:
                    break
                if var_bound_c < best_var:
                    best_var = var_bound_c
                    best_k = cand
            k = best_k
        self.k = k
        self.max_time = max_time
        self.H = ceil(log(self.max_time, self.k))
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
        if not (1 <= t <= self.max_time):
            raise ValueError(f"t must be in [1, {self.max_time}]")

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
                # skip nodes whose subtree has no leaves <= max_time
                if j * block_size >= self.max_time:
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
        max_terms = 2
        for level in range(1, self.H + 1):
            block_size = self.k ** (self.H - level)
            valid_count = ceil(self.max_time / block_size)
            # siblings per level capped by k-1
            max_terms += (min(self.k, valid_count) - 1)
        return max_terms * var_node

    def high_prob_bound(self, delta: float) -> float:
        """
        Computes a high-probability bound K such that with probability at least 1-delta,
        the noise magnitude |prefix_noise(t)| ≤ K for all t in [1, max_time],
        using a union bound and Chernoff tail on the sum of discrete Laplace terms.
        """
        if not (0 < delta < 1):
            raise ValueError("delta must be in (0, 1)")
        # Compute worst-case number of noise terms per output
        max_terms = 2
        for level in range(1, self.H + 1):
            block_size = self.k ** (self.H - level)
            valid_count = ceil(self.max_time / block_size)
            max_terms += (min(self.k, valid_count) - 1)
        T = self.max_time
        b = self.b
        # search λ in (0, -log b) to minimize the bound
        lambda_max = -np.log(b) * 0.99
        lambdas = np.linspace(1e-6, lambda_max, num=100)
        best_K = float('inf')
        for lam in lambdas:
            # For discrete-Laplace noise X, the moment-generating function is:
            #   M_X(λ) = E[e^{λX}] = (1 - b)^2 / ((1 - b e^{λ}) (1 - b e^{-λ})).
            # We decompose the denominator into:
            #   denom1 = 1 - b * e^{λ},  denom2 = 1 - b * e^{-λ}.
            # Both denom1 and denom2 must be positive for the MGF to remain finite.
            denom1 = 1 - b * np.exp(lam)
            denom2 = 1 - b * np.exp(-lam)
            if denom1 <= 0 or denom2 <= 0:
                continue
            M = (1 - b) ** 2 / (denom1 * denom2)
            # apply Chernoff: P(S ≥ K) ≤ exp(-λ K + n log M)
            # Union bound: ≤ T * exp(...)
            # set T * exp(-λ K + n log M) = delta ⇒ K = (n log M + log(T/delta)) / λ
            K_lam = (max_terms * np.log(M) + np.log(T / delta)) / lam
            if K_lam < best_K:
                best_K = K_lam
        return best_K


# Example
if __name__ == "__main__":
    MAX_TIME = 200
    tree = KaryTreeNoise(eps=1.0, max_time=MAX_TIME)

    for delta in np.geomspace(0.1,1e-20,20):
        print(f"delta ={delta}, max error={tree.high_prob_bound(delta)}")

    for t in range(1,MAX_TIME+1, max(1,MAX_TIME//10)):
        print(f"t={t:3d}, noise={tree.prefix_noise(t)}")

    for t in range(2,MAX_TIME+2, max(1,MAX_TIME//10)):
        tree = KaryTreeNoise(eps=1.0, max_time=t)
        print(f"max_time={t:3d}, H={tree.H}, scale={tree.scale}, k={tree.k}, variance={tree.variance_bound()}")
