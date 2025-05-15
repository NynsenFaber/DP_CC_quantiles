from copy import deepcopy
import numpy as np
from DP_AQ import approximate_quantiles_algo as dp_aq  # Kaplan et al. 2023
from our_mechanism.slice_quantile import SliceQuantile  # our mechanism
from analysis import get_statistics
from utils import from_rho_eps
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time


def gaussian_noise(array, bounds, scale=0.00001, seed=None):
    data = deepcopy(array)
    if seed is not None:
        np.random.seed(seed)
    return np.sort(data + np.random.normal(0, scale, len(array))), (bounds[0] - 4 * scale, bounds[1] + 4 * scale)


## Hyperparameters ##
# np.random.seed(42)  # for reproducibility
rho = 1 / 8  # privacy budget for Kaplan et al.
delta = 1e-20
eps = from_rho_eps(rho=rho, delta=delta)  # privacy budget for our mechanism
m_list = range(3, 122, 2)  # number of quantiles
num_experiments = 50
swap = False
folder_path = "../../results/working_hours_data_swap_false"
data_path = "../../data/working_hours_data.pkl"
tag = ""
alg_names = ["Kaplan et al. pure DP", "Kaplan et al. approx DP", "Slicing Quantiles g set",
             "Slicing Quantiles min g"]
title = "Working Hours"
num_algos = len(alg_names)
bounds = (0, 100)
scale_for_gaussian = 1e-5


## Load Data ##
with open(data_path, "rb") as f:
    data = pickle.load(f)
n = len(data)
print(f"Data loaded: {title} data")
print("Number of elements in data: ", len(data))
print(f"Epsilon: {eps}")
print(f"Delta: {delta}")
print("Bounds: ", bounds)

X = data
X = np.sort(X)
# add small noise to the data to ensure that they are not equal
X, bounds = gaussian_noise(X, bounds, scale=scale_for_gaussian)
g = max(scale_for_gaussian / n ** 2, np.finfo(np.float64).eps)
g_min = min(X[i] - X[i - 1] for i in range(1, n))

print(f"g_min: {g_min}")
print(f"g used: {g}")

# get equally spaced quantiles
quantiles = [np.linspace(0, 1, m + 2)[1:-1] for m in m_list]

max_errors = np.zeros((num_algos, len(m_list), num_experiments))
times = np.zeros((num_algos, len(m_list), num_experiments))
for i, m in enumerate(m_list):
    our_mechanism = SliceQuantile(bound=bounds, n=n, m=m, eps=eps, swap=swap, split=0.5, g=g)
    our_mechanism_min_g = SliceQuantile(bound=bounds, n=n, m=m, eps=eps, swap=swap, split=0.5, g=g_min)
    for j in range(num_experiments):
        # Run Kaplan et al. algorithm
        start = time.time()
        estimates = dp_aq(X, quantiles[i], bounds, eps, swap=swap)
        times[0][i][j] = time.time() - start
        statistics: dict = get_statistics(X, quantiles[i], estimates)
        max_errors[0][i][j] = statistics['max_error']

        # Run Kaplan et al. algorithm with approximate DP
        start = time.time()
        estimates = dp_aq(X, quantiles[i], bounds, rho, swap=swap, cdp=True)
        times[1][i][j] = time.time() - start
        statistics: dict = get_statistics(X, quantiles[i], estimates)
        max_errors[1][i][j] = statistics['max_error']

        # Run our mechanism
        start = time.time()
        our_estimates = our_mechanism.approximate_mechanism(X, q_list=quantiles[i], delta=delta)
        times[2][i][j] = time.time() - start
        statistics = get_statistics(X, quantiles[i], our_estimates)
        max_errors[2][i][j] = statistics['max_error']

        # Run our mechanism with min g
        start = time.time()
        our_estimates = our_mechanism_min_g.approximate_mechanism(X, q_list=quantiles[i], delta=delta)
        times[3][i][j] = time.time() - start
        statistics = get_statistics(X, quantiles[i], our_estimates)
        max_errors[3][i][j] = statistics['max_error']

# Convert to long-form DataFrame for Seaborn
records = []
for algo_idx, algo in enumerate(alg_names):
    for m_idx, m in enumerate(m_list):
        for exp in range(num_experiments):
            records.append({
                "Algorithm": algo,
                "Quantiles": m,
                "Max Error": max_errors[algo_idx, m_idx, exp],
                "Time":      times[algo_idx, m_idx, exp],
            })
df = pd.DataFrame(records)

# save dataset
import os

if not os.path.exists(folder_path):
    os.makedirs(folder_path)
with open(f"{folder_path}/data{tag}.pkl", "wb") as f:
    pickle.dump(df, f)
print("Data saved to: ", f"{folder_path}/data{tag}.pkl")

sns.set_theme(style="white")
plt.figure(figsize=(10, 6))

# Get Seaborn's default palette colors
default_colors = sns.color_palette("deep")
# red = default_colors[3]  # Red in 'deep' palette
# blue = default_colors[0]  # Blue in 'deep' palette
# green = default_colors[1]  # Green in 'deep' palette


sns.lineplot(
    data=df,
    x="Quantiles",
    y="Max Error",
    hue="Algorithm",
    errorbar=('ci', 95),  # confidence interval 95%
    # palette=[red, green, blue],
    linewidth=2,
)

plt.title(f"{title} Dataset - Utility", fontsize=16)
plt.xlabel("Number of Quantiles (m)", fontsize=14)
plt.ylabel("Max Rank Error", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.yscale("log")
plt.legend(title="Algorithm")
plt.tight_layout()
# save
plt.savefig(f"{folder_path}/utility{tag}.png", dpi=300)
print("Utility plot saved to: ", f"{folder_path}/utility{tag}.png")
plt.show()

sns.set_theme(style="white")
plt.figure(figsize=(10, 6))

# Get Seaborn's default palette colors
default_colors = sns.color_palette("deep")
red = default_colors[3]  # Red in 'deep' palette
blue = default_colors[0]  # Blue in 'deep' palette
green = default_colors[1]  # Green in 'deep' palette

df = df[~df["Algorithm"].isin(["Kaplan et al. pure DP", "Slicing Quantiles min g"])]  # remove Kaplan et al. pure DP
sns.lineplot(
    data=df,
    x="Quantiles",
    y="Max Error",
    hue="Algorithm",
    errorbar=('ci', 95),  # confidence interval 95%
    # palette=[red, blue],
    linewidth=2,
)

plt.title(f"{title} Dataset - Utility", fontsize=16)
plt.xlabel("Number of Quantiles (m)", fontsize=14)
plt.ylabel("Max Rank Error", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.yscale("log")
plt.legend(title="Algorithm")
plt.tight_layout()
# save
plt.savefig(f"{folder_path}/utility_no_pure{tag}.png", dpi=300)
print("Utility plot saved to: ", f"{folder_path}/utility_no_pure{tag}.png")
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
    # palette=[red, blue],
    linewidth=2,
)

plt.title(f"{title} - Time", fontsize=16)
plt.xlabel("Number of Quantiles (m)", fontsize=14)
plt.ylabel("Time (s)", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# plt.yscale("log")
plt.legend(title="Algorithm")
plt.tight_layout()
# save
plt.savefig(f"{folder_path}/time{tag}.png", dpi=300)
print("Time plot saved to: ", f"{folder_path}/time{tag}.png")
plt.show()