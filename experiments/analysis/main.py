import numpy as np


def get_statistics(X: np.ndarray, quantiles: np.ndarray, estimates: np.ndarray) -> float:
    """
    Compute the maximum rank error and the mean absolute rank error between the true quantiles and the estimated quantiles.
    :param X: data set
    :param quantiles: a list of true quantiles
    :param estimates: a list of estimated quantiles
    :return: the maximum rank error
    """
    # sort the data
    X = np.sort(X)

    # for each result, get the position of the closest element in the sorted array
    positions = np.searchsorted(X, estimates)
    # compute the rank error
    rank_errors = np.abs(positions / len(X) - quantiles)

    # get statistics
    max_rank_error = np.max(rank_errors)
    mean_rank_error = np.mean(rank_errors)
    output = {
        'max_rank_error': max_rank_error,
        'mean_rank_error': mean_rank_error
    }
    return output
