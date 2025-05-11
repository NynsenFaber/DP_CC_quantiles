import math
import numpy as np
from quantiles_with_continual_counting import KaryTreeNoise
from DP_AQ import single_quantile  # exponential mechanism used by Kaplan et al.
from experiments.analysis import get_statistics


def pre_process_quantiles(q_list: list[float],
                          gap: int  # TO DO: add a default value
                          ) -> list[float]:
    """
    Given a list of quantile, and a dataset size n, return a list of quantiles sufficiently spaced.

    :param q_list: list of quantiles
    :param gap: minimum gap between two quantiles
    :return: list of quantiles
    """
    ## CHECKS ##
    if len(q_list) == 0:
        raise ValueError("q_list must not be empty")
    if not all(0 <= q <= 1 for q in q_list):
        raise ValueError("All elements in q_list must be between 0 and 1")

    # sort the quantiles
    q_list = sorted(q_list)
    # remove duplicates
    q_list = list(dict.fromkeys(q_list))
    # get only the ranks that are sufficiently spaced, it gets the first one and then iteratively adds the next one
    # that is sufficiently spaced
    output: list[int] = [q_list[0]]
    for r in q_list[1:]:
        if r - output[-1] >= gap:
            output.append(r)

    return output


def get_slice_parameters(bound: tuple[float, float],
                         m: int,
                         eps: float,
                         beta: float = 0.05,  # Hyperparameter
                         g: float = 1.  # Hyperparameter
                         ) -> int:
    """
    Return the slicing parameter l, that is used to determine the size of the slices.

    :param bound: bounds of the data set tuple[float, float] (i.e. (a, b))
    :param m: number of quantiles
    :param eps: privacy budget used for the exponential mechanism
    :param beta: probability to successfully sample an interior point
    :param g: minimum gap between two numbers in the data set
    :return: An integer l that is the slicing parameter
    """
    return math.ceil((2 / eps) * np.log(m * (bound[1] - bound[0]) / (g * beta)) - 1)


def slice_quantiles(X: list,
                    q_list: list,
                    eps: float,
                    bound: tuple[float, float],
                    gamma: float = 0.001,  # TO DO: add a default value
                    verbose: bool = False,
                    split: float = 0.5,
                    swap: bool = True,
                    l: int = None,
                    seed: int = None,
                    ) -> tuple[list[float], bool]:
    """
    Compute m = len(ranks) quantiles using the SliceQuantiles algorithm. The ranks are already pre-processed to be
    sufficiently spaced using the `pre_process_quantiles` function.

    :param X: data set
    :param q_list: list of quantiles
    :param eps: privacy parameter
    :param bound: bounds of the data set tuple[float, float] (i.e. (a, b))
    :param gamma: probability of returning a random value in (a, b) instead of the estimate.
    :param verbose: if True, print information about the algorithm
    :param split: Is the fraction of privacy budget to give to continual counting noise
    :param swap: if True, bounded DP else unbounded DP
    :param l: `slicing parameter`, it affects the accurcay of the interior point mechanism (the exponential mechanism). It
    is used to determine the size of the slices. If None, it is computed using the `get_slice_parameters` function.
    :param seed: seed for the random number generator

    :return: a list of points in (a, b) that are the estimate of the ranks. It returns also a boolean that indicates
    if the algorithm returned a random value or not.
    """
    ## CHECKS ##
    if len(X) == 0:
        raise ValueError("X must not be empty")
    if len(q_list) == 0:
        raise ValueError("q_list must not be empty")
    if eps < 0:
        raise ValueError("eps must be greater than 0")
    if bound[0] >= bound[1]:
        raise ValueError("bound must be (a, b) with a < b")
    if split <= 0 or split >= 1:
        raise ValueError("split must be in (0, 1)")

    if seed is not None:
        np.random.seed(seed)

    m = len(q_list)
    n = len(X)
    output = np.zeros(m)  # to store the output
    eps_2 = eps * split  # privacy budget for the continual counting noise
    eps_1 = eps * (1 - split)  # privacy budget for the exponential mechanism

    # if swap:
    #     eps_1 = eps_1 / 4
    # else:
    #     eps_1 = eps_1 / 2
    eps_1 = eps_1 / 2

    ranks: list[int] = [math.floor(q * n) for q in q_list]  # get ranks
    X = sorted(X)  # sort the data

    # Get slice parameter
    if l is None:
        l = get_slice_parameters(bound, m, eps_1, g=min(X[i] - X[i - 1] for i in range(1, n)))

    # Add Countinual Counting noise to the ranks
    tree = KaryTreeNoise(eps=eps_2, max_time=m)
    ranks = [rank + tree.prefix_noise(i) for i, rank in enumerate(ranks, start=1)]

    ## Handling edge cases ##
    # flip coin c with probability gamma
    c = np.random.uniform(0, 1)
    head_c = c < gamma
    # look if at least two ranks are close to each other
    flag_spacing = any(ranks[i] - ranks[i - 1] < 2 * l + 1 for i in range(1, m))
    # look for at leasr noe rank is out of bounds
    flag_bounds = any(rank < l or rank > n - l - 1 for rank in ranks)
    if head_c or flag_spacing or flag_bounds:
        output = np.random.uniform(bound[0], bound[1], size=m)
        return output, True

    ## Create the slices
    slices = {}
    for i in range(m):  # create slices only for the ranks in [0, n].
        left_index = max(ranks[i] - l, 0)
        right_index = min(ranks[i] + l + 1, n)
        slices[i] = X[left_index:right_index + 1]  # +1 because the right bound is exclusive

    ## check if the slices intersect
    if verbose:
        count = 0
        for i in range(len(slices) - 1):
            if slices[i][-1] > slices[i + 1][0]:
                count += 1
        print("Number of slices that intersect: ", count)

    ## The accuracy of the exponential may depend in practice on the order we answer the queries.
    ## From first to last ##
    left_bound = bound[0]
    for i in range(m):
        output[i] = single_quantile(slices[i],
                                    bounds=(left_bound, bound[1]),
                                    quantile=0.5,
                                    epsilon=eps_1,
                                    swap=True,
                                    seed=seed)
        # update the left bound
        left_bound = output[i]

    return output, False


if __name__ == "__main__":
    m = 10
    q_list = np.linspace(0, 1, m + 2)[1:-1]
    n = 100000
    gap = 0.001
    q_list = pre_process_quantiles(q_list, gap)
    print("Ranks used: ", q_list)

    # Generate a random dataset
    a = 0
    b = 2 ** 32
    X = np.random.uniform(a, b, n)
    eps_1 = 0.3
    eps_2 = 0.7
    bound = (a, b)
    l = 40
    prob_random = 0.0
    estimates, _ = slice_quantiles(X, q_list, l, eps_1 + eps_2, bound, prob_random, verbose=True)
    print("Estimates: ", estimates)
    statistics = get_statistics(X, q_list, estimates)
    for key, value in statistics.items():
        print(f"{key}: {value}")
