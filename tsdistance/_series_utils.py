import numpy as np


def as_float_array(ts):
    return np.asarray(ts, dtype=float).reshape(-1)


def as_series_matrix(ts):
    array = np.asarray(ts, dtype=float)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    return array


def resolve_window(lenx, leny, w):
    return max(lenx, leny) if w is None else int(w)


def window_slice(length, center, w):
    start = max(0, center - w)
    stop = min(length, center + w + 1)
    if start >= stop:
        anchor = min(max(center, 0), length - 1)
        return anchor, anchor + 1
    return start, stop


def window_minmax(series, center, w):
    start, stop = window_slice(len(series), center, w)
    window = series[start:stop]
    return float(np.min(window)), float(np.max(window))
