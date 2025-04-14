import numpy as np
from .continual_counting import pure_continual_counting


def kaplan_exp_mech(arr, eps, a, b, q):
    q *= len(arr)
    itv_lengths = []
    endpts = [a]
    for val in arr:
        itv_lengths.append(val - endpts[-1])
        endpts.append(val)
    itv_lengths.append(b - endpts[-1])
    endpts.append(b)
    utis = np.arange(0, len(itv_lengths))
    utis = -np.abs(utis - (q + 0.5))
    probs = np.array(itv_lengths) * np.exp(utis * eps / 2)
    #return probs
    probs = probs.cumsum()
    rand = np.random.uniform(0, probs[-1])
    idx = np.searchsorted(probs, rand)
    elem = np.random.uniform(endpts[idx], endpts[idx+1])
    return elem


def our_dp_rec(arr, eps, a, b, quantiles, width):
    #print("Range")
    #print(a,b)
    #print(quantiles)
    if len(quantiles) == 0:
        return []
    m = len(quantiles) // 2
    #print("Quantile chosen")
    #print(quantiles[m])
    l = int(quantiles[m] * len(arr)) - width
    r = int(quantiles[m] * len(arr)) + width
    #print("Arr vals")
    #print(arr[l], arr[r])
    est = kaplan_exp_mech(arr[l:r+1], eps, a, b, 0.5)
    #print("Q_EST")
    #print(est)
    L = []
    if m > 0:
        L = our_dp_rec(arr, eps, a, est, quantiles[:m], width)
    R = []
    if m+1 < len(quantiles):
        R = our_dp_rec(arr, eps, est, b, quantiles[m+1:], width)
    return L + [est] + R


def our_dp_top(arr, eps, B, quantiles, split=0.5):
    eps1, eps2 = (eps * split, eps * (1-split))
    quantiles.sort()
    quantiles = np.array(quantiles)
    noise = pure_continual_counting(len(quantiles), eps1)
    # print("Noise Vals")
    # print(noise)
    noise = noise.astype('float64') / len(arr)
    quantiles += noise
    width = int( 2/eps2 * np.log(20 * B * len(quantiles)) ) + 1
    # print('Width: %d' % width)
    # print("Quantiles:")
    # print(quantiles)
    assert quantiles[0] >= (width + 10) / len(arr)
    assert quantiles[-1] <= 1-(width+10) / len(arr)
    for i in range(len(quantiles)-1):
        assert quantiles[i+1] - quantiles[i] >= (width+10) / len(arr)
    return our_dp_rec(arr, eps2, 1, B, quantiles, width)
