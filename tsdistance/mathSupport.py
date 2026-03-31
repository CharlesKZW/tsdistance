import numpy as np


def _as_float_array(ts):
    return np.asarray(ts, dtype=float).reshape(-1)


def make_envelopes(ts, w):
    series = _as_float_array(ts)
    length = series.shape[0]
    if length == 0:
        return np.array([]), np.array([])
    if w is None:
        w = length
    w = int(w)
    upper = np.empty(length, dtype=float)
    lower = np.empty(length, dtype=float)
    for idx in range(length):
        start = max(0, idx - w)
        stop = min(length, idx + w + 1)
        window = series[start:stop]
        upper[idx] = np.max(window)
        lower[idx] = np.min(window)
    return upper, lower


def MakeEnvForSingleTS(ts, w):
    return make_envelopes(ts, w)


def lower_b(ts, w):
    return make_envelopes(ts, w)[1]


def lower_b_n(ts, w):
    return lower_b(ts, w)


def lcss_subcost(x, y, epsilon):
    return 1 if abs(x - y) <= epsilon else 0


def swale_subcost(x, y, epsilon, p, r):
    return 0 if abs(x - y) <= epsilon else p


def msm_dist(new_point, x_prev, y_curr, c):
    if (x_prev <= new_point <= y_curr) or (x_prev >= new_point >= y_curr):
        return c
    return c + min(abs(new_point - x_prev), abs(new_point - y_curr))


def edr_subcost(x, y, epsilon):
    return 0 if abs(x - y) <= epsilon else 1


def dev(ts):
    series = np.asarray(ts, dtype=float)
    if series.ndim == 1:
        series = series.reshape(1, -1)
    if series.shape[1] < 3:
        raise ValueError("Derivative-based measures require at least 3 time points.")
    derivative = np.empty((series.shape[0], series.shape[1] - 2), dtype=float)
    for row in range(series.shape[0]):
        for idx in range(1, series.shape[1] - 1):
            derivative[row, idx - 1] = (
                (series[row, idx] - series[row, idx - 1])
                + (series[row, idx + 1] - series[row, idx - 1]) / 2.0
            ) / 2.0
    return derivative
