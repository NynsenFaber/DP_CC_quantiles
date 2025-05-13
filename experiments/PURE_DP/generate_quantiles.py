import numpy as np


# def generate_random_quantiles()
#     # Generate quantiles that are sufficiently spaced
#     quantiles = []
#     for i, m in tqdm.tqdm(enumerate(m_list)):
#         # Check if m is feasible for the given gap
#         if m * gap > 1:
#             print(f"m={m} is too large for the given gap={gap}. Skipping remaining values.")
#             break
#
#         count = 0  # Count how many realizations of quantiles we have
#         quantile_combinations = []
#         stop = 0
#         while count < num_experiments and stop < 10:
#             stop += 1
#             np.random.seed(seed + count)
#             # Generate random quantiles and sort them
#             random_quantiles = np.sort(
#                 np.random.uniform(low=0, high=1, size=np.random.randint(low=m, high=min(n, m + 10))))
#             # Post-process quantiles to ensure sufficient spacing
#             processed_quantiles = pre_process_quantiles(random_quantiles, gap=gap)
#             if len(processed_quantiles) >= m:
#                 # Take all possible combinations of m quantiles
#                 combinations = list(itertools.combinations(processed_quantiles, m))
#                 # take at most num_experiments - count combinations
#                 quantile_combinations.extend(combinations[:num_experiments - count])
#                 count += len(combinations[:num_experiments - count])
#
#         quantiles.append(quantile_combinations[:num_experiments])
#
#     quantiles = np.array(quantiles, dtype=object)


def equally_spaced_qantiles(m: int):
    # Generate equally spaced quantiles
    return np.linspace(0, 1, m + 2)[1:-1]  # Exclude 0 and 1
