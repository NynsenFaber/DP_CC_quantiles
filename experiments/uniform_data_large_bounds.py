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
import time


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
m_list = range(10, 210, 10)  # number of quantiles
num_algos = 2  # Kaplan et al. and our mechanism
num_experiments = 100
swap = False

## Load Data ##
with open("../data/uniform_data_large_bounds.pkl", "rb") as f:
    data = pickle.load(f)
# print("Data loaded: working_hours_data.pkl (Adult Hours)")
print("Data loaded: Uniform data (Large bounds)")
print("Number of elements in data: ", len(data["data"]))
n = min(n, len(data["data"]))
print(f"Sampled {n} elements")
# sample n random elements
X = np.random.choice(data["data"], n, replace=False)

bounds = data["bounds"]
print("Bounds: ", bounds)
# add small noise to the data to ensure that they are not equal
X, bounds = gaussian_noise(X, bounds)

# get equally spaced quantiles
quantiles = [equally_spaced_qantiles(m) for m in m_list]

max_errors = np.zeros((num_algos, len(m_list), num_experiments))
times = np.zeros((num_algos, len(m_list), num_experiments))
returned_random = np.zeros((num_algos, len(m_list)))
for i, m in enumerate(m_list):
    for j in range(num_experiments):
        seed = seed + j
        # Run Kaplan et al. algorithm
        start = time.time()
        estimates = dp_aq(X, quantiles[i], bounds, eps, swap=swap, seed=seed)
        times[0][i][j] = time.time() - start
        statistics: dict = get_statistics(X, quantiles[i], estimates)
        max_errors[0][i][j] = statistics['max_error']

        # Run our mechanism
        start = time.time()
        our_estimates, flag_random = slice_quantiles(X, q_list=quantiles[i], eps=eps, bound=bounds, split=0.5,
                                                     seed=seed, swap=swap, continual_counting="k-ary")
        times[1][i][j] = time.time() - start
        returned_random[1][i] += int(flag_random)
        if not flag_random:
            statistics = get_statistics(X, quantiles[i], our_estimates)
            max_errors[1][i][j] = statistics['max_error']

# Convert to long-form DataFrame for Seaborn
records = []
for algo_idx, algo in enumerate(["Kaplan et al.", "Our Mechanism"]):
    for m_idx, m in enumerate(m_list):
        for exp in range(num_experiments):
            records.append({
                "Algorithm":       algo,
                "Quantiles":       m,
                "Max Error":       max_errors[algo_idx, m_idx, exp],
                "Returned Random": returned_random[algo_idx][m_idx] / num_experiments,
                "Time":            times[algo_idx, m_idx, exp],
            })
df = pd.DataFrame(records)
df_our_mechanism = df[df['Algorithm'] == 'Our Mechanism']

# save dataset
folder_path = "../results/uniform_data_large_bounds_swap_false/"
import os
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
with open(f"{folder_path}/data.pkl", "wb") as f:
    pickle.dump(df, f)

sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))

# Get Seaborn's default palette colors
default_colors = sns.color_palette("deep")
red = default_colors[3]   # Red in 'deep' palette
blue = default_colors[0]  # Blue in 'deep' palette

sns.lineplot(
    data=df,
    x="Quantiles",
    y="Max Error",
    hue="Algorithm",
    errorbar=('ci', 95),  # confidence interval 95%
    palette=[red, blue],
    linewidth=2,
)

plt.title("Uniform Dataset - Utility", fontsize=16)
plt.xlabel("Number of Quantiles (m)", fontsize=14)
plt.ylabel("Max Rank Error", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# plt.yscale("log")
plt.legend(title="Algorithm")
plt.tight_layout()
# save
plt.savefig(f"{folder_path}/utility.png", dpi=300)
plt.show()

sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))

sns.barplot(
    data=df_our_mechanism,
    x="Quantiles",
    y="Returned Random",
    linewidth=2,
)

plt.title("Uniform Dataset - Fraction of Success", fontsize=16)
plt.xlabel("Number of Quantiles (m)", fontsize=14)
plt.ylabel("Fraction", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
# save
plt.savefig(f"{folder_path}/fraction_success.png", dpi=300)
plt.show()

# Make a plot for the time
sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))

sns.lineplot(
    data=df,
    x="Quantiles",
    y="Time",
    hue="Algorithm",
    errorbar=('ci', 95),  # confidence interval 95%
    palette=[red, blue],
    linewidth=2,
)

plt.title("Uniform Dataset - Time", fontsize=16)
plt.xlabel("Number of Quantiles (m)", fontsize=14)
plt.ylabel("Time (s)", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# plt.yscale("log")
plt.legend(title="Algorithm")
plt.tight_layout()
# save
plt.savefig(f"{folder_path}/time.png", dpi=300)
plt.show()
