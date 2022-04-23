import numpy as np
from scipy.stats import kendalltau

# Spearman Rank Correlation coefficient rs
def spearman(R, Q):
    N = len(R)
    denominator = N*(N**2-1)
    numerator = 6*sum((R-Q)**2)
    rS = 1-(numerator/denominator)
    return rS

# Weighted Spearman Rank Correlation coefficient rw
def weighted_spearman(R, Q):
    N = len(R)
    denominator = N**4 + N**3 - N**2 - N
    numerator = 6 * sum((R - Q)**2 * ((N - R + 1) + (N - Q + 1)))
    rW = 1 - (numerator / denominator)
    return rW