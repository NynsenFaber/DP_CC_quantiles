import numpy as np


def load_data(data_dist: str,
              bounds: tuple[float, float],
              n: int,
              seed: int = None,
              **kwargs
              ) -> np.ndarray:
    """
    Sample data from a given distribution.

    :param data_dist: type of distribution to sample from
    :param bounds: bounds of the data
    :param n: number of samples to generate
    :param seed: seed for the random number generator
    :param kwargs: additional arguments for the distribution

    :return: An array of floats sampled from the specified distribution.
    """
    if seed is not None:
        np.random.seed(seed)

    B = bounds[1] - bounds[0]
    if data_dist == 'uniform':
        arr = np.array(np.random.uniform(bounds[0], bounds[1], n))

    elif data_dist == 'gaussian':
        # sample from a normal distribution with mean B/2 and std B/10
        arr = np.array(np.random.normal(B / 2, B / 10, n))
        # clip the values to be within the bounds
        arr = np.maximum(bounds[0], np.minimum(bounds[1], arr))

    elif data_dist == 'mixture':
        # sample from a mixture of gaussians
        m = kwargs.get('m', 10)  # number of gaussians
        width = kwargs.get('width', 100)  # space between the gaussians
        # selects m random centers in the range [20 * width, bounds[1] - 20 * width]
        centers = np.random.uniform(20 * width, bounds[1] - 20 * width, m)
        arr = np.array([])
        num_left = n
        for i in range(0, m - 1):
            # sample from a gaussian with mean centers[i] and std width n // m elements
            data = np.random.normal(centers[i], width, n // m)
            # apply bounds
            data = np.maximum(bounds[0], np.minimum(bounds[1], data))
            # add the elements to the array
            arr = np.concatenate((data, arr))
            num_left -= n // m
        data = np.random.normal(centers[-1], width, num_left)
        # apply bounds
        data = np.maximum(bounds[0], np.minimum(bounds[1], data))
        # add the elements to the array
        arr = np.concatenate((data, arr))
    else:
        raise Exception('Unsupported Data Distribution')
    return arr