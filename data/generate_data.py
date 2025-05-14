import numpy as np
import pickle
from tqdm import tqdm


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
        centers = np.random.uniform(bounds[0], bounds[1], m - 1)
        centers = np.concatenate((centers, np.array([bounds[0], bounds[1]])))  # two centers at the bounds
        arr = np.array([])
        num_left = n
        for i in range(0, m - 1):
            # sample from a gaussian with mean centers[i] and std width n // m elements
            data = np.random.normal(centers[i], width, size=n // m)
            # add the elements to the array
            arr = np.concatenate((data, arr))
            num_left -= n // m
        data = np.random.normal(centers[-1], width, num_left)
        # add the elements to the array
        arr = np.concatenate((data, arr))
        # CHECK FOR DUPLICATES
        arr = np.unique(arr)
    else:
        raise Exception('Unsupported Data Distribution')

    return arr


if __name__ == "__main__":
    np.random.seed(42)
    # ## Generate uniform data like Kaplan et al.
    # n = 1_000_000
    # bounds = (-5, 5)
    # data = np.random.uniform(bounds[0], bounds[1], n)
    # output = {"bounds": bounds,
    #           "data":   data,
    #           "type":   "uniform"}
    # # save as pickle
    # with open('uniform_data_small_bounds.pkl', 'wb') as f:
    #     pickle.dump(output, f)
    # print("Data saved to uniform_data_small_bounds.pkl")
    #
    # ## Generate Uniform data with large bounds
    # bounds = (0, 2 ** 32)
    # data = np.random.uniform(bounds[0], bounds[1], n)
    # output = {"bounds": bounds,
    #           "data":   data,
    #           "type":   "uniform"}
    # # save as pickle
    # with open('uniform_data_large_bounds.pkl', 'wb') as f:
    #     pickle.dump(output, f)
    # print("Data saved to uniform_data_large_bounds.pkl")
    #
    # ## Generate Gaussian data like Kaplan et al.
    # mean = 0
    # std = np.sqrt(5)
    # data = np.random.normal(mean, std, n)
    # output = {"bounds": (-10 * std, 10 * std),  # 10 std is a practical bound
    #           "data":   data,
    #           "type":   "gaussian"}
    # # save as pickle
    # with open('gaussian_data_small_bounds.pkl', 'wb') as f:
    #     pickle.dump(output, f)
    # print("Data saved to gaussian_data_small_bounds.pkl")
    #
    # ## Generate Gaussian data with large bounds
    # mean = 0
    # std = np.sqrt(2 ** 32)
    # data = np.random.normal(mean, std, n)
    # output = {"bounds": (-10 * std, 10 * std),  # 10 std is a practical bound
    #           "data":   data,
    #           "type":   "gaussian"}
    # # save as pickle
    # with open('gaussian_data_large_bounds.pkl', 'wb') as f:
    #     pickle.dump(output, f)
    # print("Data saved to gaussian_data_large_bounds.pkl")

    ## Generate Mixture data
    n = 100_000
    # n_min = 200_000
    width = 10
    bound_list = [(-2 ** i + 10 * width, 2 ** i - 10 * width) for i in range(20, 41)]
    # min_gap = 1
    output = []
    for bound in tqdm(bound_list):
        data = load_data('mixture', bound, n, m=1_000, width=width)
        data = np.sort(data)
        # index_to_remove = []
        # for i in range(len(data) - 1):
        #     if data[i + 1] - data[i] < min_gap:
        #         index_to_remove.append(i + 1)
        # data = np.delete(data, index_to_remove)
        # while len(data) < n_min:
        #     print("adding")
        #     # add more data
        #     additional_data = load_data('mixture', bound, n_min - len(data), m=10000, width=width)
        #     data = np.concatenate((data, additional_data))
        #     index_to_remove = []
        #     for i in range(len(data) - 1):
        #         if data[i + 1] - data[i] < min_gap:
        #             index_to_remove.append(i + 1)
        #     data = np.delete(data, index_to_remove)

        # Store
        output.append({"bounds": (bound[0] - 10 * width, bound[1] + 10 * width),
                       "data":   data,
                       "type":   "mixture"})
    # save as pickle
    with open('mixture_data_large_bounds.pkl', 'wb') as f:
        pickle.dump(output, f)
    print("Data saved to mixture_data_large_bounds.pkl")
