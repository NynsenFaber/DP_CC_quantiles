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
    :param n: size of the dataset
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


def slice_quantiles(X: list,
                    q_list: list,
                    l: int,
                    eps_1: float,
                    eps_2: float,
                    bound: tuple[float, float],
                    prob_random: float = 0.0,  # TO DO: add a default value
                    verbose: bool = False,
                    ) -> list[float]:
    """
    Compute m = len(ranks) quantiles using the SliceQuantiles algorithm. The ranks are already pre-processed to be
    sufficiently spaced using the `pre_process_quantiles` function.

    :param X: data set
    :param q_list: list of quantiles
    :param l: `slicing parameter`, it affects the accurcay of the interior point mechanism (the exponential mechanism). It
    is used to determine the size of the slices.
    :param eps_1: privacy parameter for the countinual counting mechanism
    :param eps_2: privacy parameter for the exponential mechanism
    :param bound: bounds of the data set tuple[float, float] (i.e. (a, b))
    :param prob_random: probability of returning a random value in (a, b) instead of the estimate.
    :param verbose: if True, print information about the algorithm

    :return: a list of points in (a, b) that are the estimate of the ranks
    """
    ## CHECKS ##
    if len(X) == 0:
        raise ValueError("X must not be empty")
    if len(q_list) == 0:
        raise ValueError("q_list must not be empty")
    if l < 1:
        raise ValueError("l must be greater than 1")
    if eps_1 < 0:
        raise ValueError("eps_1 must be greater than 0")
    if eps_2 < 0:
        raise ValueError("eps_2 must be greater than 0")
    if bound[0] >= bound[1]:
        raise ValueError("bound must be (a, b) with a < b")

    m = len(q_list)
    n = len(X)
    output = np.zeros(m)  # to store the output
    # get ranks
    ranks: list[int] = [math.floor(q * n) for q in q_list]
    # sort the data
    X = sorted(X)
    # Add Countinual Counting noise to the ranks
    tree = KaryTreeNoise(eps=eps_1, max_time=m)
    ranks = [rank + tree.prefix_noise(i) for i, rank in enumerate(ranks, start=1)]

    ## handling edge cases
    # If some rank is negative or greater than n, return random values
    I = []  # store the index of the good ranks
    for i in range(m):
        if ranks[i] < 0 or ranks[i] > n:
            output[i] = np.random.uniform(bound[0], bound[1])
        else:
            I.append(i)

    ## Create the slices
    slices = {}
    for i in I:  # create slices only for the ranks in [0, n].
        left_index = max(ranks[i] - l, 0)
        right_index = min(ranks[i] + l + 1, n)
        slices[i] = X[left_index:right_index + 1]  # +1 because the right bound is exclusive

    # check if the slices intersect
    if verbose:
        count = 0
        for i in range(len(I) - 1):
            if slices[I[i]][-1] > slices[I[i + 1]][0]:
                count += 1
        print("Number of slices that intersect: ", count)

    ## The accuracy of the exponential may depend in practice on the order we answer the queries.
    ## From first to last ##
    left_bound = bound[0]
    for i in I:
        # sample a random coin with head probability prob_random
        if np.random.uniform(0, 1) < prob_random:
            # return a random value in (a, b)
            output[i] = np.random.uniform(bound[0], bound[1])
            # do not update the left bound as it is not informative
        else:
            output[i] = single_quantile(slices[i],
                                        bounds=(left_bound, bound[1]),
                                        quantile=0.5,
                                        epsilon=eps_2,
                                        swap=True)
            # update the left bound
            left_bound = output[i]

    return output


if __name__ == "__main__":
    m = 10
    q_list = np.linspace(0, 1, m + 2)[1:-1]
    n = 1000
    gap = 0.001
    q_list = pre_process_quantiles(q_list, gap)
    print("Ranks used: ", q_list)

    # Generate a random dataset
    a = 0
    b = 2**32
    X = np.random.uniform(a, b, n)
    eps_1 = 1.
    eps_2 = 1.
    bound = (a, b)
    l = 40
    prob_random = 0.0
    estimates = slice_quantiles(X, q_list, l, eps_1, eps_2, bound, prob_random, verbose=True)
    print("Estimates: ", estimates)
    statistics = get_statistics(X, q_list, estimates)
    for key, value in statistics.items():
        print(f"{key}: {value}")
