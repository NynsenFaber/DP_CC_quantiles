import numpy as np


def from_rho_eps(rho: float, delta: float) -> float:
    """
    Convert rho to epsilon with fixed delta.
    :param rho: privacy budget for zCDP
    :param delta: failure probability of pure DP
    """
    return rho + 2 * np.sqrt(rho * np.log(1 / delta))


def get_rho(eps: float, delta: float) -> float:
    """
    Convert epsilon to rho with fixed delta.
    :param eps: privacy budget for pure DP
    :param delta: failure probability of pure DP
    """
    return np.sqrt(np.log(1 / delta) + eps) - np.sqrt(np.log(1 / delta))

def get_eps_from_rho_delta(rho: float, delta: float) -> float:
    """
    Lemma 3.7 https://arxiv.org/pdf/1605.02065
    :param rho:
    :param delta:
    :return:
    """
    return rho - 5 * (4 * rho) ** (1/4) + 2 * np.sqrt(rho * np.log(1 / delta))


