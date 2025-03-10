import numpy as np
from scipy import ks_2samp

# calculating the weighted cdf by summing over the weights
def weighted_ecdf(values, weights):
    """
    Compute the weighted empirical CDF.
    Returns:
        x_sorted : sorted values
        cdf      : cumulative distribution (from 0 to 1)
    """
    # Sort by the values
    order = np.argsort(values)
    x_sorted = values[order]
    w_sorted = weights[order]

    # Weighted cumulative sum
    cum_w = np.cumsum(w_sorted)
    total_w = cum_w[-1]

    # Weighted CDF from 0.0 to 1.0
    cdf = cum_w / total_w
    return x_sorted, cdf

# using the function above, do the ks test
def weighted_ks_2samp(x1, w1, x2, w2):
    """
    Weighted two-sample Kolmogorov-Smirnov test (no p-value).
    Returns:
        ks_stat : the maximum distance between the two weighted CDFs
    """
    x1_sorted, cdf1 = weighted_ecdf(x1, w1)
    x2_sorted, cdf2 = weighted_ecdf(x2, w2)

    i1 = 0
    i2 = 0
    d_max = 0.0
    n1 = len(x1_sorted)
    n2 = len(x2_sorted)

    # Current cdf values
    c1 = 0.0
    c2 = 0.0

    # Merge the two sorted arrays
    while i1 < n1 and i2 < n2:
        v1 = x1_sorted[i1]
        v2 = x2_sorted[i2]
        if v1 < v2:
            c1 = cdf1[i1]
            i1 += 1
        elif v2 < v1:
            c2 = cdf2[i2]
            i2 += 1
        else:
            c1 = cdf1[i1]
            c2 = cdf2[i2]
            i1 += 1
            i2 += 1
        d = abs(c1 - c2)
        if d > d_max:
            d_max = d

    # If one array is exhausted, continue stepping through the other
    while i1 < n1:
        c1 = cdf1[i1]
        i1 += 1
        d = abs(c1 - c2)
        if d > d_max:
            d_max = d

    while i2 < n2:
        c2 = cdf2[i2]
        i2 += 1
        d = abs(c1 - c2)
        if d > d_max:
            d_max = d

    return d_max  # no p-value for the weighted case

