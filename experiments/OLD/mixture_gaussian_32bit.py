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
n = 50_000  # max number of elements to sample
np.random.seed(42)  # for reproducibility
rho = 1 / 10  # privacy budget for Kaplan et al.
delta = 1e-6
eps = from_rho_eps(rho=rho, delta=delta)  # privacy budget for our mechanism
m_list = range(10, 130, 10)  # number of quantiles
num_algos = 4  # Kaplan et al. and our mechanism
num_experiments = 100
swap = False
data_path = "../../data/mixture_data_large_bounds.pkl"
data_index = 12  # 32 bit is the last one
tag = "_32bit"
folder_path = f"../results/mixture_gaussian{tag}/"
alg_names = ["Kaplan et al. pure DP", "Kaplan et al. approx DP", "Slicing Quantiles g=1", "Slicing Quantiles min g"]

## Load Data ##
with open(data_path, "rb") as f:
    data = pickle.load(f)
data = data[data_index] # last is for 32 bit
# print("Data loaded: working_hours_data.pkl (Adult Hours)")
print("Data loaded: Gaussian data (Small bounds)")
print("Number of elements in data: ", len(data["data"]))
n = min(n, len(data["data"]))
print(f"Sampled {n} elements")
print("Rho: ", rho)
print(f"Epsilon: {eps}")
print(f"Delta: {delta}")
# sample n random elements
X = np.random.choice(data["data"], n, replace=False)

bounds = data["bounds"]
print("Bounds: ", bounds)
# add small noise to the data to ensure that they are not equal
X, bounds = gaussian_noise(X, bounds)
g = 1.  # try this
min_g = min(X[i] - X[i - 1] for i in range(1, n))

# get equally spaced quantiles
quantiles = [np.linspace(0, 1, m + 2)[1:-1] for m in m_list]

max_errors = np.zeros((num_algos, len(m_list), num_experiments))
times = np.zeros((num_algos, len(m_list), num_experiments))
for i, m in enumerate(m_list):
    our_mechanism = SliceQuantile(bound=bounds, n=n, m=m, eps=eps, swap=swap, split=0.5, g=g)
    our_mechanism_min_g = SliceQuantile(bound=bounds, n=n, m=m, eps=eps, swap=swap, split=0.5, g=min_g)
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
        # compute minimum spacing between points
        our_estimates = our_mechanism.approximate_mechanism(X, q_list=quantiles[i], delta=delta)
        times[2][i][j] = time.time() - start
        statistics = get_statistics(X, quantiles[i], our_estimates)
        max_errors[2][i][j] = statistics['max_error']

        # Run our mechanism
        start = time.time()
        # compute minimum spacing between points
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

sns.set_theme(style="whitegrid")
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

plt.title("Mixture Guassian Dataset - Utility", fontsize=16)
plt.xlabel("Number of Quantiles (m)", fontsize=14)
plt.ylabel("Max Rank Error", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# plt.yscale("log")
plt.legend(title="Algorithm")
plt.tight_layout()
# save
plt.savefig(f"{folder_path}/utility{tag}.png", dpi=300)
print("Utility plot saved to: ", f"{folder_path}/utility{tag}.png")
plt.show()

sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))

# Get Seaborn's default palette colors
default_colors = sns.color_palette("deep")
red = default_colors[3]  # Red in 'deep' palette
blue = default_colors[0]  # Blue in 'deep' palette
green = default_colors[1]  # Green in 'deep' palette

df = df[df["Algorithm"] != "Kaplan et al. pure DP"]  # remove Kaplan et al. pure DP
sns.lineplot(
    data=df,
    x="Quantiles",
    y="Max Error",
    hue="Algorithm",
    errorbar=('ci', 95),  # confidence interval 95%
    # palette=[red, blue],
    linewidth=2,
)

plt.title("Mixture Guassian Dataset - Utility", fontsize=16)
plt.xlabel("Number of Quantiles (m)", fontsize=14)
plt.ylabel("Max Rank Error", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# plt.yscale("log")
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

plt.title("Mixture Guassian Dataset - Time", fontsize=16)
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
