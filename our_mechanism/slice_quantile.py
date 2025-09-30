import math
import numpy as np
from quantiles_with_continual_counting import KaryTreeNoise
from DP_AQ import single_quantile  # exponential mechanism used by Kaplan et al.


def get_slice_parameter(bound: tuple[float, float],
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


class SliceQuantile:

    def __init__(self,
                 bound: tuple[float, float],
                 n: int,
                 m: int,
                 eps: float,
                 split: float = 0.5,
                 swap: bool = False,
                 l: int = None,
                 g: float = None,
                 ) -> None:
        """
        :param eps: privacy parameter
        :param bound: bounds of the data set tuple[float, float] (i.e. (a, b))
        :param g: minimum gap between two numbers in the data set. If None, it is set to 1.0
        :param n: size of the data set
        :param m: number of quantiles to compute
        :param split: Is the fraction of privacy budget to give to continual counting noise
        :param swap: if True, bounded DP else unbounded DP
        :param l: `slicing parameter`, it affects the accurcay of the interior point mechanism (the exponential mechanism).
        It is the siz of the slices. If None, it is computed using the `get_slice_parameters` function.
        """
        self.m = m  # number of quantiles
        self.n = n  # size of the data set
        self.bound = bound  # bounds of the data set
        self.eps = eps  # total privacy budget
        self.eps_cc, self.eps_em = get_epsilons(eps, split, swap)  # privacy budgets for the two mechanisms
        self.split = split  # fraction of privacy budget for continual counting noise
        self.swap = swap  # if True, bounded DP else unbounded DP
        if l is None:
            self.g = g if g is not None else 1.0
            self.l = get_slice_parameter(bound=bound,
                                         m=m,
                                         eps=self.eps_em,
                                         g=self.g)
        # instatiate the k-ary tree with eps_cc privacy budget
        self.tree = KaryTreeNoise(eps=self.eps_cc, max_time=m)

    def is_delta_approximate_DP(self, delta: float, q_list: list[float]) -> bool:
        """
        Check if the algorithm is delta approximate DP. It uses the fact that the k-ary tree is a binary tree.
        """
        ranks = [0] + [int(self.n * q) for q in q_list] + [self.n]
        eta = min(np.diff(ranks))
        return self.tree.high_prob_bound(delta=delta) < 0.5 * (eta - 1) - self.l

    def get_min_delta(self, q_list: list[float]) -> float:
        """
        It returns the minimum delta that is needed to guarantee that the algorithm is delta approximate DP.

        :param q_list: list of quantiles
        :return: the minimum delta
        """

        # Perform a binary search on the exponent of delta
        low, high = -100, 0  # Exponents for delta (10^low to 10^high)
        while high - low > 0.5:  # Precision for the exponent
            mid = (low + high) / 2
            delta = 10 ** mid  # Convert exponent to delta
            if self.is_delta_approximate_DP(delta, q_list):
                high = mid
            else:
                low = mid
        return 10 ** high  # Return the minimum delta

    def approximate_mechanism(self,
                              X: list,
                              q_list: list,
                              delta: float,
                              verbose: bool = False,
                              ) -> list[float]:
        """
        Compute m = len(ranks) quantiles using the SliceQuantiles algorithm.

        :param X: data set
        :param q_list: list of quantiles
        :param delta: upper bound on approximate DP
        :param verbose: if True, print information about the algorithm

        :return: a list of points in (a, b) that are the estimate of the ranks. It returns also a boolean that indicates
        if the algorithm returned a random value or not.
        """
        if len(q_list) != self.m:
            raise ValueError("q_list must have the same length as m")
        if not self.is_delta_approximate_DP(delta, q_list):
            raise ValueError("The algorithm is not delta approximate DP")

        ranks: list[int] = [math.floor(q * self.n) for q in q_list]  # get ranks
        X = sorted(X)  # sort the data

        # Add Countinual Counting noise to the ranks
        ranks = [rank + self.tree.prefix_noise(i) for i, rank in enumerate(ranks, start=1)]

        ## Create the slices
        slices = []
        for i in range(self.m):  # create slices only for the ranks in [0, n].
            left_index = max(ranks[i] - self.l, 0)
            right_index = min(ranks[i] + self.l + 1, self.n)
            slices.append(X[left_index:right_index + 1])  # +1 because the right bound is exclusive

        ## check if the slices intersect
        if verbose:
            count = 0
            for i in range(len(slices) - 1):
                if slices[i][-1] > slices[i + 1][0]:
                    count += 1
            if count > 0: print("Number of slices that intersect: ", count)

        def algo_helper(slices, bound):
            if len(slices) == 0:
                return []
            elif len(slices) == 1:
                return [single_quantile(slices[0], bound, 0.5, epsilon=self.eps_em, swap=True)]
            len_slice = len(slices)
            array = slices[len_slice // 2]  # get the middle slice
            z = single_quantile(array, bound, 0.5, epsilon=self.eps_em, swap=True)
            a, b = bound
            return (algo_helper(slices[:len_slice // 2], (a, z))
                    + [z]
                    + algo_helper(slices[len_slice // 2 + 1:], (z, b))
                    )

        return algo_helper(slices, self.bound)


def get_statistics(X: np.ndarray, quantiles: np.ndarray, estimates: np.ndarray) -> float:
    """
    Compute the maximum rank error and the mean absolute rank error between the true quantiles and the estimated quantiles.
    :param X: data set
    :param quantiles: a list of quantiles (values in [0, 1])
    :param estimates: a list of estimated qth-quantiles (values in X)
    :return: a dictionary with the maximum rank error and the mean absolute rank error
    """
    # sort the data
    X = np.sort(X)
    n = len(X)

    # for each result, get the position of the closest element in the sorted array
    positions = np.searchsorted(X, estimates)
    true_positions = [math.floor(q * n) for q in quantiles]

    # compute the normalized rank error
    rank_errors = np.abs(positions - true_positions)

    # get statistics
    max_rank_error = np.max(rank_errors)
    mean_rank_error = np.mean(rank_errors)
    output = {
        'max_error':  max_rank_error,
        'mean_error': mean_rank_error  # Kaplan et al. uses this
    }
    return output


if __name__ == "__main__":
    n = 100_000
    m = 50
    q_list = np.linspace(0, 1, m + 2)[1:-1]

    # Generate a random dataset
    a = 0
    b = 2 ** 32
    X = np.random.uniform(a, b, n)
    eps = 1.
    bound = (a, b)
    mechanism = SliceQuantile(bound=bound, n=n, m=m, eps=eps, swap=False, split=0.5)
    estimates = mechanism.approximate_mechanism(X, q_list, delta=1e-10, verbose=True)
    print("Estimates: ", estimates)
    statistics = get_statistics(X, q_list, estimates)
    for key, value in statistics.items():
        print(f"{key}: {value}")
