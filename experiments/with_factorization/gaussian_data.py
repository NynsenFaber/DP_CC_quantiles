from copy import deepcopy

import numpy as np
from DP_AQ import approximate_quantiles_algo as dp_aq  # Kaplan et al. 2023
from our_mechanism.slice_quantile import pre_process_quantiles, slice_quantiles  # our mechanism
from analysis import get_statistics
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from generate_quantiles import equally_spaced_qantiles
import itertools
import tqdm


def gaussian_noise(array, bounds, scale=0.00001, seed=None):
    data = deepcopy(array)
    if seed is not None:
        np.random.seed(seed)
    return np.sort(data + np.random.normal(0, scale, len(array))), (bounds[0] - 4 * scale, bounds[1] + 4 * scale)


## Hyperparameters ##
n = 250_000  # max number of elements to sample
seed = 42  # for reproducibility
np.random.seed(seed)
eps = 1.  # privacy budget
m_list = range(10, 150, 10)  # number of quantiles
num_algos = 3  # Kaplan et al. and our mechanism
num_experiments = 100
swap = True

## Load Data ##
with open("../data/gaussian_data_small_bounds.pkl", "rb") as f:
    data = pickle.load(f)
# print("Data loaded: working_hours_data.pkl (Adult Hours)")
print("Data loaded: Gaussian data (Large bounds)")
print("Number of elements in data: ", len(data["data"]))
n = min(n, len(data["data"]))
print(f"Sampled {n} elements")
# sample n random elements
X = np.random.choice(data["data"], n, replace=False)

bounds = data["bounds"]
# add small noise to the data to ensure that they are not equal
X, bounds = gaussian_noise(X, bounds)

# get equally spaced quantiles
quantiles = [equally_spaced_qantiles(m) for m in m_list]

max_errors = np.zeros((num_algos, len(m_list), num_experiments))
returned_random = np.zeros((num_algos, len(m_list)))
for i, m in enumerate(m_list):
    for j in range(num_experiments):
        seed = seed + j
        # Run Kaplan et al. algorithm
        estimates = dp_aq(X, quantiles[i], bounds, eps, swap=swap, seed=seed)
        statistics: dict = get_statistics(X, quantiles[i], estimates)
        max_errors[0][i][j] = statistics['max_error']

        # Run our mechanism
        our_estimates, flag_random = slice_quantiles(X, q_list=quantiles[i], eps=eps, bound=bounds, split=0.5,
                                                     seed=seed, swap=swap, continual_counting="k-ary")
        returned_random[1][i] += int(flag_random)
        if not flag_random:
            statistics = get_statistics(X, quantiles[i], our_estimates)
            max_errors[1][i][j] = statistics['max_error']

        # Run our mechanism
        our_estimates, flag_random = slice_quantiles(X, q_list=quantiles[i], eps=eps, bound=bounds, split=0.8,
                                                     seed=seed, swap=swap, continual_counting="factorization")
        returned_random[2][i] += int(flag_random)
        if not flag_random:
            statistics = get_statistics(X, quantiles[i], our_estimates)
            max_errors[2][i][j] = statistics['max_error']

# Run Kaplan et al. algorithm with no swap

# Convert to long-form DataFrame for Seaborn
records = []
for algo_idx, algo in enumerate(["Kaplan et al.", "Our Mechanism, k-ary", "Our Mechanism, factorization"]):
    for m_idx, m in enumerate(m_list):
        for exp in range(num_experiments):
            records.append({
                "Algorithm":       algo,
                "Quantiles":       m,
                "Max Error":       max_errors[algo_idx, m_idx, exp],
                "Returned Random": returned_random[algo_idx][m_idx] / num_experiments,
            })
df = pd.DataFrame(records)
df_our_mechanism = df[df['Algorithm'] == 'Our Mechanism, k-ary']
df_our_mechanism_factorization = df[df['Algorithm'] == 'Our Mechanism, factorization']

sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))

# Get Seaborn's default palette colors
default_colors = sns.color_palette("deep")
red = default_colors[3]   # Red in 'deep' palette
blue = default_colors[0]  # Blue in 'deep' palette
green = default_colors[1] # Green in 'deep' palette

sns.lineplot(
    data=df,
    x="Quantiles",
    y="Max Error",
    hue="Algorithm",
    errorbar=('ci', 95),  # confidence interval 95%
    palette=[red, blue, green],
    linewidth=2,
)

plt.title("Guassian Dataset", fontsize=16)
plt.xlabel("Number of Quantiles (m)", fontsize=12)
plt.ylabel("Max Rank Error", fontsize=12)
# plt.yscale("log")
plt.legend(title="Algorithm")
plt.tight_layout()
plt.show()

sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))

sns.barplot(
    data=df_our_mechanism,
    x="Quantiles",
    y="Returned Random",
    linewidth=2,
)

plt.title("Guassian Dataset - Fraction of Random Values Returned - K-ary", fontsize=16)
plt.xlabel("Number of Quantiles (m)", fontsize=12)
plt.ylabel("Fraction", fontsize=12)
plt.tight_layout()
plt.show()

sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))

sns.barplot(
    data=df_our_mechanism_factorization,
    x="Quantiles",
    y="Returned Random",
    linewidth=2,
)

plt.title("Guassian Dataset - Fraction of Random Values Returned - Factorization", fontsize=16)
plt.xlabel("Number of Quantiles (m)", fontsize=12)
plt.ylabel("Fraction", fontsize=12)
plt.tight_layout()
plt.show()
