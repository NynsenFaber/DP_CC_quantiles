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

## Hyperparameters ##
seed = 42  # for reproducibility
np.random.seed(seed)
B_list = [50_000, 100_000, 150_000, 200_000, 250_000, 300_000, 350_000, 400_000, 450_000, 500_000]
eps = 1.  # privacy budget
m = 100  # unique nnumber of quantiles
num_algos = 2  # Kaplan et al. and our mechanism
num_experiments = 100
swap = True

# get equally spaced quantiles
quantiles = equally_spaced_qantiles(m)

max_errors = np.zeros((num_algos, len(B_list), num_experiments))
times = np.zeros((num_algos, len(B_list), num_experiments))
returned_random = np.zeros((num_algos, len(B_list)))
for i, B in enumerate(B_list):
    X = range(B)
    bounds = (0, B)
    for j in range(num_experiments):
        seed = seed + j
        # Run Kaplan et al. algorithm
        start = time.time()
        estimates = dp_aq(X, quantiles, bounds, eps, swap=swap, seed=seed)
        times[0][i][j] = time.time() - start
        statistics: dict = get_statistics(X, quantiles, estimates)
        max_errors[0][i][j] = statistics['max_error']

        # Run our mechanism
        start = time.time()
        our_estimates, flag_random = slice_quantiles(X, q_list=quantiles, eps=eps, bound=bounds, split=0.5,
                                                     seed=seed, swap=swap, continual_counting="k-ary")
        times[1][i][j] = time.time() - start
        returned_random[1][i] += int(flag_random)
        if not flag_random:
            statistics = get_statistics(X, quantiles, our_estimates)
            max_errors[1][i][j] = statistics['max_error']

# Convert to long-form DataFrame for Seaborn
records = []
for algo_idx, algo in enumerate(["Kaplan et al.", "Our Mechanism"]):
    for B_idx, B in enumerate(B_list):
        for exp in range(num_experiments):
            records.append({
                "Algorithm":       algo,
                "B":               B,
                "Max Error":       max_errors[algo_idx, B_idx, exp],
                "Returned Random": returned_random[algo_idx][B_idx] / num_experiments,
                "Time":            times[algo_idx, B_idx, exp],
            })
df = pd.DataFrame(records)
df_our_mechanism = df[df['Algorithm'] == 'Our Mechanism']

# save dataset
folder_path = "../../results/integer_with_increasing_bounds/"
import os

if not os.path.exists(folder_path):
    os.makedirs(folder_path)
with open(f"{folder_path}/data.pkl", "wb") as f:
    pickle.dump(df, f)

sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))

# Get Seaborn's default palette colors
default_colors = sns.color_palette("deep")
red = default_colors[3]  # Red in 'deep' palette
blue = default_colors[0]  # Blue in 'deep' palette

sns.lineplot(
    data=df,
    x="B",
    y="Max Error",
    hue="Algorithm",
    errorbar=('ci', 95),  # confidence interval 95%
    palette=[red, blue],
    linewidth=2,
)

plt.title("Uniform Dataset - Utility", fontsize=16)
plt.xlabel("B - size of data domain", fontsize=14)
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
    x="B",
    y="Returned Random",
    linewidth=2,
)

plt.title("Uniform Dataset - Fraction of Success", fontsize=16)
plt.xlabel("B - size of data domain", fontsize=14)
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
    x="B",
    y="Time",
    hue="Algorithm",
    errorbar=('ci', 95),  # confidence interval 95%
    palette=[red, blue],
    linewidth=2,
)

plt.title("Uniform Dataset - Time", fontsize=16)
plt.xlabel("B - size of data domain", fontsize=14)
plt.ylabel("Time (s)", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# plt.yscale("log")
plt.legend(title="Algorithm")
plt.tight_layout()
# save
plt.savefig(f"{folder_path}/time.png", dpi=300)
plt.show()
