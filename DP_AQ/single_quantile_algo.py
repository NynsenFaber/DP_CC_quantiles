import numpy as np


def racing_sample(log_terms, seed=None):
    """Numerically stable method for sampling from an exponential distribution.

    Args:
      log_terms: Array of terms of form log(coefficient) - (exponent term).

    Returns:
      A sample from the exponential distribution determined by terms. See
      Algorithm 1 from the paper "Duff: A Dataset-Distance-Based
      Utility Function Family for the Exponential Mechanism"
      (https://arxiv.org/pdf/2010.04235.pdf) for details; each element of terms is
      analogous to a single log(lambda(A_k)) - (eps * k/2) in their algorithms.
    """
    if seed is not None:
        np.random.seed(seed)
    return np.argmin(np.log(np.log(1.0 / np.random.uniform(size=log_terms.shape))) - log_terms)


## MOD: added seed for reproducibility
def single_quantile(sorted_array, bounds, quantile, epsilon, swap, seed=None):
    a, d = bounds
    n = len(sorted_array)
    sorted_array = np.clip(sorted_array, a, d)
    sorted_array = np.concatenate(([a], sorted_array, [d]))
    intervals = np.diff(sorted_array)
    sensitivity = max(quantile, 1 - quantile)
    if swap:
        sensitivity = 1.0

    utility = -np.abs(np.arange(0, n + 1) - (quantile * n))
    with np.errstate(divide='ignore'):  # MOD: add this line to ignore divide by zero
        idx_left = racing_sample(np.log(intervals) + (epsilon / (2.0 * sensitivity) * utility), seed=seed)
    if seed is not None:
        np.random.seed(seed)
    return np.random.uniform(sorted_array[idx_left], sorted_array[idx_left + 1])
