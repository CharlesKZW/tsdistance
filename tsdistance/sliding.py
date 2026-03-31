import numpy as np

from ._series_utils import as_float_array


def _cross_correlation(x, y):
    x_arr = as_float_array(x)
    y_arr = as_float_array(y)
    return np.correlate(x_arr, y_arr, mode="full")


def NCC(x, y):
    r"""Return the full cross-correlation sequence."""

    return _cross_correlation(x, y)


def NCCb(x, y):
    r"""Return the biased normalized cross-correlation sequence."""

    x_arr = as_float_array(x)
    y_arr = as_float_array(y)
    denom = float(max(len(x_arr), len(y_arr)))
    if denom == 0:
        return np.array([])
    return _cross_correlation(x_arr, y_arr) / denom


def NCCc(x, y):
    r"""Return the coefficient normalized cross-correlation sequence."""

    x_arr = as_float_array(x)
    y_arr = as_float_array(y)
    denom = np.linalg.norm(x_arr) * np.linalg.norm(y_arr)
    if denom == 0:
        return np.zeros(len(x_arr) + len(y_arr) - 1, dtype=float)
    return _cross_correlation(x_arr, y_arr) / denom


def NCCu(x, y):
    r"""Return the unbiased normalized cross-correlation sequence."""

    x_arr = as_float_array(x)
    y_arr = as_float_array(y)
    corr = _cross_correlation(x_arr, y_arr)
    overlap = np.correlate(
        np.ones(len(x_arr), dtype=float), np.ones(len(y_arr), dtype=float), mode="full"
    )
    return corr / overlap
