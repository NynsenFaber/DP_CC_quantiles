import math
import numpy as np
from quantiles_with_continual_counting import KaryTreeNoise
from DP_AQ import single_quantile  # exponential mechanism used by Kaplan et al.
from experiments.analysis import get_statistics


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
    return math.ceil((2 / eps) * np.log(4 * m * (bound[1] - bound[0]) / (g * beta)) - 1)


def get_epsilons(eps: float, split: float, swap: bool) -> tuple[float, float]:
    eps_cc = eps * split  # privacy budget for the continual counting noise
    eps_em = eps * (1 - split)  # privacy budget for the exponential mechanism
    if swap:
        eps_cc = eps_cc / 2
        eps_em = eps_em / 3
    else:
        eps_em = eps_em / 2
    return eps_cc, eps_em


def approximate_slice_quantiles(X: list,
                                q_list: list,
                                eps: float,
                                bound: tuple[float, float],
                                g=None,
                                verbose: bool = False,
                                split: float = 0.5,
                                swap: bool = False,
                                l: int = None,
                                ) -> list[float]:
    """
    Compute m = len(ranks) quantiles using the SliceQuantiles algorithm. The ranks are already pre-processed to be
    sufficiently spaced using the `pre_process_quantiles` function.

    :param X: data set
    :param q_list: list of quantiles
    :param eps: privacy parameter
    :param bound: bounds of the data set tuple[float, float] (i.e. (a, b))
    :param g: minimum gap between two numbers in the data set. If None, it is computed directly from the data set.
    :param verbose: if True, print information about the algorithm
    :param split: Is the fraction of privacy budget to give to continual counting noise
    :param swap: if True, bounded DP else unbounded DP
    :param l: `slicing parameter`, it affects the accurcay of the interior point mechanism (the exponential mechanism). It
    is used to determine the size of the slices. If None, it is computed using the `get_slice_parameters` function.

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

    m = len(q_list)
    n = len(X)

    ranks: list[int] = [math.floor(q * n) for q in q_list]  # get ranks
    X = sorted(X)  # sort the data

    # Get the privacy budgets
    eps_cc, eps_em = get_epsilons(eps=eps, split=split, swap=swap)

    # Get slice parameter
    if l is None:
        if g is None:
            g = min(X[i] - X[i - 1] for i in range(1, n))
        l = get_slice_parameters(bound, m, eps_em, g=g)

    # Add Countinual Counting noise to the ranks
    tree = KaryTreeNoise(eps=eps_cc, max_time=m)
    ranks = [rank + tree.prefix_noise(i) for i, rank in enumerate(ranks, start=1)]

    ## Create the slices
    slices = []
    for i in range(m):  # create slices only for the ranks in [0, n].
        left_index = max(ranks[i] - l, 0)
        right_index = min(ranks[i] + l + 1, n)
        slices.append(X[left_index:right_index + 1])  # +1 because the right bound is exclusive

    ## check if the slices intersect
    if verbose:
        count = 0
        for i in range(len(slices) - 1):
            if slices[i][-1] > slices[i + 1][0]:
                count += 1
        print("Number of slices that intersect: ", count)

    def algo_helper(slices, bounds):
        if len(slices) == 0:
            return []
        elif len(slices) == 1:
            return [single_quantile(slices[0], bounds, 0.5, epsilon=eps_em, swap=True)]
        len_slice = len(slices)
        array = slices[len_slice // 2]
        z = single_quantile(array, bounds, 0.5, epsilon=eps_em, swap=True)
        a, b = bounds
        return (algo_helper(slices[:len_slice // 2], (a, z))
                + [z]
                + algo_helper(slices[len_slice // 2 + 1:], (z, b))
                )

    return algo_helper(slices, bound)


if __name__ == "__main__":
    n = 100_000
    m = 10
    q_list = np.linspace(0, 1, m + 2)[1:-1]

    # Generate a random dataset
    a = 0
    b = 2 ** 32
    X = np.random.uniform(a, b, n)
    eps = 1.
    bound = (a, b)
    estimates = approximate_slice_quantiles(X, q_list, eps, bound, verbose=True, split=0.5, swap=False)
    print("Estimates: ", estimates)
    statistics = get_statistics(X, q_list, estimates)
    for key, value in statistics.items():
        print(f"{key}: {value}")
