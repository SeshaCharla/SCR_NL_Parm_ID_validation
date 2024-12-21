import numpy as np

def calc_eta(x1, u1):
    """Recursively calculates eta based on x1 and u1"""
    n = len(x1)
    eta = np.zeros(n)
    for i in range(1, n):
        eta[i] = u1[i-1] - x1[i]
    return eta

def calc_eta_TD(x1, u1, tskips):
    n = len(x1)
    eta = np.zeros(n)
    for i in range(len(tskips)-1):
        eta[tskips[i]:tskips[i+1]] = calc_eta(x1[tskips[i]:tskips[i+1]], u1[tskips[i]:tskips[i+1]])
    return eta
