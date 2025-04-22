import numpy as np
from generate_data import load_data
from DP_AQ import approximate_quantiles_algo as dp_aq
from our_mechanism import our_dp_top
from analysis import get_statistics

n = 10_000
B = 2**32
bounds = (1, B)
eps = 1.

X = load_data('gaussian', bounds, n, seed=42)
X = load_data('mixture', bounds, n, seed=420)
quantiles = np.linspace(0, 1, 21)[1:-1]
print(quantiles)

estimates = dp_aq(X, quantiles, bounds, eps, swap=True)
statistics: dict = get_statistics(X, quantiles, estimates)
for key, value in statistics.items():
    print(f"{key}: {value}")

our_estimates = our_dp_top(X, eps, B, quantiles, split=0.7)
statistics = get_statistics(X, quantiles, our_estimates)
for key, value in statistics.items():
    print(f"{key}: {value}")
