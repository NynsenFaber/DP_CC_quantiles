def slice_quantiles(X: list,
                    q_list: list,
                    l: int,
                    eps: float,
                    bound: tuple[float, float],
                    gamma: float = 0.001,  # TO DO: add a default value
                    verbose: bool = False,
                    split: float = 0.5,
                    ) -> tuple[list[float], bool]:
    """
    Compute m = len(ranks) quantiles using the SliceQuantiles algorithm. The ranks are already pre-processed to be
    sufficiently spaced using the `pre_process_quantiles` function.

    :param X: data set
    :param q_list: list of quantiles
    :param l: `slicing parameter`, it affects the accurcay of the interior point mechanism (the exponential mechanism). It
    is used to determine the size of the slices.
    :param eps: privacy parameter
    :param bound: bounds of the data set tuple[float, float] (i.e. (a, b))
    :param gamma: probability of returning a random value in (a, b) instead of the estimate.
    :param verbose: if True, print information about the algorithm
    :param split: Is the fraction of privacy budget to give to continual counting noise

    :return: a list of points in (a, b) that are the estimate of the ranks. It returns also a boolean that indicates
    if the algorithm returned a random value or not.
    """
    ## CHECKS ##
    if len(X) == 0:
        raise ValueError("X must not be empty")
    if len(q_list) == 0:
        raise ValueError("q_list must not be empty")
    if l < 1:
        raise ValueError("l must be greater than 1")
    if eps < 0:
        raise ValueError("eps must be greater than 0")
    if bound[0] >= bound[1]:
        raise ValueError("bound must be (a, b) with a < b")
    if split <= 0 or split >= 1:
        raise ValueError("split must be in (0, 1)")

    m = len(q_list)
    n = len(X)
    output = np.zeros(m)  # to store the output
    eps_1 = eps * split  # privacy budget for the continual counting noise
    eps_2 = eps * (1 - split)  # privacy budget for the exponential mechanism
    ranks: list[int] = [math.floor(q * n) for q in q_list]  # get ranks
    X = sorted(X)  # sort the data

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
        if np.random.uniform(0, 1) < gamma:
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


def pure_slice_quantiles(X: list,
                         q_list: list,
                         eps: float,
                         bound: tuple[float, float],
                         gamma: float = 0.001,  # TO DO: add a default value
                         verbose: bool = False,
                         split: float = 0.5,
                         swap: bool = True,
                         l: int = None,
                         continual_counting: str = "factorization",
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
    :param continual_counting: type of continual counting to use. It can be "factorization" or "k-ary".

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
    output = np.zeros(m)  # to store the output

    eps_cc = eps * split  # privacy budget for the continual counting noise
    eps_em = eps * (1 - split)  # privacy budget for the exponential mechanism
    if swap:
        eps_cc = eps_cc / 2
        eps_em = eps_em / 3
    else:
        eps_em = eps_em / 2

    ranks: list[int] = [math.floor(q * n) for q in q_list]  # get ranks
    X = sorted(X)  # sort the data

    # Get slice parameter
    if l is None:
        l = get_slice_parameters(bound, m, eps_em, g=min(X[i] - X[i - 1] for i in range(1, n)))

    if continual_counting == "k-ary":
        # Add Countinual Counting noise to the ranks
        tree = KaryTreeNoise(eps=eps_cc, max_time=m)
        ranks = [rank + tree.prefix_noise(i) for i, rank in enumerate(ranks, start=1)]
    elif continual_counting == "factorization":
        noise = pure_continual_counting(m=m, eps=eps_cc)
        ranks = [round(rank + noise[i]) for i, rank in enumerate(ranks)]
    else:
        raise ValueError("continual_counting must be 'factorization' or 'k-ary'")

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
        return output

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
                                    epsilon=eps_em,
                                    swap=True)
        # update the left bound
        left_bound = output[i]

    return output