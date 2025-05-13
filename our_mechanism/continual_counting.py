import numpy as np
from scipy import linalg


def pure_continual_counting(m: int, eps: float) -> np.ndarray:
    """
    Generate noise for the pure continual counting mechanism.
    :param m: number of noise values to generate
    :param eps: privacy budget
    :return: a numpy array of noise values
    """
    mat = np.array([[1] * k + [0] * (m - k) for k in range(1, m + 1)])
    L = linalg.sqrtm(mat)
    width = 1 / eps * linalg.norm(L, ord=1)
    noise = np.random.laplace(0, width, m)
    noise = L @ noise
    return noise


def test():
    m = 10
    eps = 1.
    noise = pure_continual_counting(m, eps)
    print("Noise Vals")
    print(noise)


if __name__ == "__main__":
    test()
