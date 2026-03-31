import math

import numpy as np

from ._series_utils import as_float_array, resolve_window, window_minmax
from .mathSupport import dev, make_envelopes, msm_dist


def _constraint_mode(constraint):
    if constraint in (None, "None"):
        return None
    if constraint == "Sakoe-Chiba":
        return "Sakoe-Chiba"
    raise ValueError("constraint must be None or 'Sakoe-Chiba'")


def _dispatch_fast(fast, fast_impl, slow_impl, *args):
    return fast_impl(*args) if fast else slow_impl(*args)


def _derivative_series(series):
    return np.squeeze(dev(np.array([series], dtype=float)))


def _dtw_root_distance(x, y, w=None):
    x_arr = as_float_array(x)
    y_arr = as_float_array(y)
    window = resolve_window(len(x_arr), len(y_arr), w)
    cost = np.full((len(x_arr) + 1, len(y_arr) + 1), np.inf)
    cost[0, 0] = 0.0
    for i in range(1, len(x_arr) + 1):
        j_start = max(1, i - window)
        j_stop = min(len(y_arr), i + window)
        for j in range(j_start, j_stop + 1):
            point = (x_arr[i - 1] - y_arr[j - 1]) ** 2
            cost[i, j] = point + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    return math.sqrt(cost[len(x_arr), len(y_arr)])


def _lb_keogh_square(x, upper, lower):
    sumd = 0.0
    for idx, value in enumerate(x):
        if value > upper[idx]:
            sumd += (value - upper[idx]) ** 2
        elif value < lower[idx]:
            sumd += (value - lower[idx]) ** 2
    return sumd


def _lcss_distance_impl(x, y, epsilon, w=None):
    x_arr = as_float_array(x)
    y_arr = as_float_array(y)
    if len(x_arr) == 0 or len(y_arr) == 0:
        return 1.0
    window = resolve_window(len(x_arr), len(y_arr), w)
    cost = np.zeros((len(x_arr) + 1, len(y_arr) + 1), dtype=float)
    for i in range(1, len(x_arr) + 1):
        j_start = max(1, i - window)
        j_stop = min(len(y_arr), i + window)
        for j in range(j_start, j_stop + 1):
            if abs(x_arr[i - 1] - y_arr[j - 1]) <= epsilon:
                cost[i, j] = cost[i - 1, j - 1] + 1.0
            else:
                cost[i, j] = max(cost[i - 1, j], cost[i, j - 1])
    return 1.0 - (cost[len(x_arr), len(y_arr)] / min(len(x_arr), len(y_arr)))


def _edr_distance_impl(x, y, epsilon, w=None):
    x_arr = as_float_array(x)
    y_arr = as_float_array(y)
    window = resolve_window(len(x_arr), len(y_arr), w)
    cost = np.full((len(x_arr) + 1, len(y_arr) + 1), np.inf)
    cost[:, 0] = np.arange(len(x_arr) + 1, dtype=float)
    cost[0, :] = np.arange(len(y_arr) + 1, dtype=float)
    for i in range(1, len(x_arr) + 1):
        j_start = max(1, i - window)
        j_stop = min(len(y_arr), i + window)
        for j in range(j_start, j_stop + 1):
            subcost = 0.0 if abs(x_arr[i - 1] - y_arr[j - 1]) <= epsilon else 1.0
            cost[i, j] = min(
                cost[i - 1, j - 1] + subcost,
                cost[i - 1, j] + 1.0,
                cost[i, j - 1] + 1.0,
            )
    return float(cost[len(x_arr), len(y_arr)])


def _erp_distance_impl(x, y, m, w=None):
    x_arr = as_float_array(x)
    y_arr = as_float_array(y)
    window = resolve_window(len(x_arr), len(y_arr), w)
    cost = np.full((len(x_arr) + 1, len(y_arr) + 1), np.inf)
    cost[0, 0] = 0.0
    for i in range(1, len(x_arr) + 1):
        cost[i, 0] = cost[i - 1, 0] + abs(x_arr[i - 1] - m)
    for j in range(1, len(y_arr) + 1):
        cost[0, j] = cost[0, j - 1] + abs(y_arr[j - 1] - m)
    for i in range(1, len(x_arr) + 1):
        j_start = max(1, i - window)
        j_stop = min(len(y_arr), i + window)
        for j in range(j_start, j_stop + 1):
            cost[i, j] = min(
                cost[i - 1, j - 1] + abs(x_arr[i - 1] - y_arr[j - 1]),
                cost[i - 1, j] + abs(x_arr[i - 1] - m),
                cost[i, j - 1] + abs(y_arr[j - 1] - m),
            )
    return float(cost[len(x_arr), len(y_arr)])


def _twed_distance_impl(x, timesx, y, timesy, lamb, nu, w=None):
    x_arr = as_float_array(x)
    y_arr = as_float_array(y)
    tx = as_float_array(timesx)
    ty = as_float_array(timesy)
    if len(x_arr) != len(tx) or len(y_arr) != len(ty):
        raise ValueError("Times and series must have matching lengths.")
    window = resolve_window(len(x_arr), len(y_arr), w)
    xp = np.concatenate(([0.0], x_arr))
    yp = np.concatenate(([0.0], y_arr))
    txp = np.concatenate(([0.0], tx))
    typ = np.concatenate(([0.0], ty))
    cost = np.full((len(x_arr) + 1, len(y_arr) + 1), np.inf)
    cost[0, 0] = 0.0
    for i in range(1, len(x_arr) + 1):
        cost[i, 0] = (
            cost[i - 1, 0]
            + abs(xp[i] - xp[i - 1])
            + nu * abs(txp[i] - txp[i - 1])
            + lamb
        )
    for j in range(1, len(y_arr) + 1):
        cost[0, j] = (
            cost[0, j - 1]
            + abs(yp[j] - yp[j - 1])
            + nu * abs(typ[j] - typ[j - 1])
            + lamb
        )
    for i in range(1, len(x_arr) + 1):
        j_start = max(1, i - window)
        j_stop = min(len(y_arr), i + window)
        for j in range(j_start, j_stop + 1):
            delete_x = (
                cost[i - 1, j]
                + abs(xp[i] - xp[i - 1])
                + nu * abs(txp[i] - txp[i - 1])
                + lamb
            )
            delete_y = (
                cost[i, j - 1]
                + abs(yp[j] - yp[j - 1])
                + nu * abs(typ[j] - typ[j - 1])
                + lamb
            )
            match = (
                cost[i - 1, j - 1]
                + abs(xp[i] - yp[j])
                + abs(xp[i - 1] - yp[j - 1])
                + nu * (abs(txp[i] - typ[j]) + abs(txp[i - 1] - typ[j - 1]))
            )
            cost[i, j] = min(delete_x, delete_y, match)
    return float(cost[len(x_arr), len(y_arr)])


def _msm_distance_impl(x, y, c, w=None):
    x_arr = as_float_array(x)
    y_arr = as_float_array(y)
    if len(x_arr) == 0 or len(y_arr) == 0:
        raise ValueError("MSM requires non-empty time series.")
    window = resolve_window(len(x_arr), len(y_arr), w)
    cost = np.full((len(x_arr), len(y_arr)), np.inf)
    cost[0, 0] = abs(x_arr[0] - y_arr[0])
    for i in range(1, len(x_arr)):
        if i <= window:
            cost[i, 0] = cost[i - 1, 0] + msm_dist(x_arr[i], x_arr[i - 1], y_arr[0], c)
    for j in range(1, len(y_arr)):
        if j <= window:
            cost[0, j] = cost[0, j - 1] + msm_dist(y_arr[j], x_arr[0], y_arr[j - 1], c)
    for i in range(1, len(x_arr)):
        j_start = max(1, i - window)
        j_stop = min(len(y_arr) - 1, i + window)
        for j in range(j_start, j_stop + 1):
            cost[i, j] = min(
                cost[i - 1, j - 1] + abs(x_arr[i] - y_arr[j]),
                cost[i - 1, j] + msm_dist(x_arr[i], x_arr[i - 1], y_arr[j], c),
                cost[i, j - 1] + msm_dist(y_arr[j], x_arr[i], y_arr[j - 1], c),
            )
    return float(cost[len(x_arr) - 1, len(y_arr) - 1])


def _swale_distance_impl(x, y, p, r, epsilon, w=None):
    x_arr = as_float_array(x)
    y_arr = as_float_array(y)
    window = resolve_window(len(x_arr), len(y_arr), w)
    cost = np.full((len(x_arr) + 1, len(y_arr) + 1), np.inf)
    cost[:, 0] = np.arange(len(x_arr) + 1, dtype=float) * p
    cost[0, :] = np.arange(len(y_arr) + 1, dtype=float) * p
    for i in range(1, len(x_arr) + 1):
        j_start = max(1, i - window)
        j_stop = min(len(y_arr), i + window)
        for j in range(j_start, j_stop + 1):
            if abs(x_arr[i - 1] - y_arr[j - 1]) <= epsilon:
                cost[i, j] = cost[i - 1, j - 1] + r
            else:
                cost[i, j] = min(cost[i - 1, j], cost[i, j - 1]) + p
    return float(cost[len(x_arr), len(y_arr)])


def _lb_keogh_lcss_impl(y, x, epsilon, w=None):
    y_arr = as_float_array(y)
    x_arr = as_float_array(x)
    if len(x_arr) == 0 or len(y_arr) == 0:
        return 1.0
    window = resolve_window(len(x_arr), len(y_arr), w)
    matches = 0
    for idx, value in enumerate(y_arr):
        lower, upper = window_minmax(x_arr, idx, window)
        if lower - epsilon <= value <= upper + epsilon:
            matches += 1
    return 1.0 - (matches / min(len(x_arr), len(y_arr)))


def _lb_keogh_erp_impl(y, x, m, w=None):
    y_arr = as_float_array(y)
    x_arr = as_float_array(x)
    window = resolve_window(len(x_arr), len(y_arr), w)
    lb_dist = 0.0
    for idx, value in enumerate(y_arr):
        lower, upper = window_minmax(x_arr, idx, window)
        upper = max(m, upper)
        lower = min(m, lower)
        if value > upper:
            lb_dist += value - upper
        elif value < lower:
            lb_dist += lower - value
    return lb_dist


def _lb_kim_erp_impl(y, x, m):
    y_arr = as_float_array(y)
    x_arr = as_float_array(x)
    x_max = max(m, float(np.max(x_arr)))
    x_min = min(m, float(np.min(x_arr)))
    y_max = max(m, float(np.max(y_arr)))
    y_min = min(m, float(np.min(y_arr)))
    return max(
        abs(x_arr[0] - y_arr[0]),
        abs(x_arr[-1] - y_arr[-1]),
        abs(x_max - y_max),
        abs(x_min - y_min),
    )


def _lb_twed_impl(y, x, lamb, nu, w=None):
    y_arr = as_float_array(y)
    x_arr = as_float_array(x)
    window = resolve_window(len(x_arr), len(y_arr), w)
    lb_dist = min(
        abs(x_arr[0] - y_arr[0]),
        abs(x_arr[0]) + nu + lamb,
        abs(y_arr[0]) + nu + lamb,
    )
    for idx in range(1, len(y_arr)):
        lower, upper = window_minmax(x_arr, idx, window)
        upper = max(upper, y_arr[idx - 1])
        lower = min(lower, y_arr[idx - 1])
        if y_arr[idx] > upper:
            lb_dist += min(nu, y_arr[idx] - upper)
        elif y_arr[idx] < lower:
            lb_dist += min(nu, lower - y_arr[idx])
    return float(lb_dist)


def _lb_msm_impl(y, x, c, w=None):
    y_arr = as_float_array(y)
    x_arr = as_float_array(x)
    window = resolve_window(len(x_arr), len(y_arr), w)
    lb_dist = abs(x_arr[0] - y_arr[0])
    for idx in range(1, len(y_arr)):
        lower, upper = window_minmax(x_arr, idx, window)
        if y_arr[idx] > upper and y_arr[idx - 1] >= y_arr[idx]:
            lb_dist += min(abs(y_arr[idx] - upper), c)
        elif y_arr[idx] < lower and y_arr[idx - 1] <= y_arr[idx]:
            lb_dist += min(abs(y_arr[idx] - lower), c)
    return float(lb_dist)


def _lb_keogh(y, x, w=None):
    y_arr = as_float_array(y)
    x_arr = as_float_array(x)
    window = resolve_window(len(x_arr), len(y_arr), w)
    lb_dist = 0.0
    for idx, value in enumerate(y_arr):
        lower, upper = window_minmax(x_arr, idx, window)
        if value > upper:
            lb_dist += (value - upper) ** 2
        elif value < lower:
            lb_dist += (value - lower) ** 2
    return math.sqrt(lb_dist)


def ddtw(x, y, constraint=None, w=None):
    mode = _constraint_mode(constraint)
    if mode is None:
        return ddtw_n(x, y)
    return ddtw_scb(x, y, w)


def ddtw_n(x, y):
    return _dtw_root_distance(_derivative_series(x), _derivative_series(y))


def ddtw_scb(x, y, w):
    return _dtw_root_distance(_derivative_series(x), _derivative_series(y), w)


def lb_improved(x, y, w=None, fast=True):
    return _dispatch_fast(fast, lb_improved_numba, lb_improved_n, x, y, w)


def lb_improved_numba(x, y, w=None):
    return lb_improved_n(x, y, w)


def lb_improved_n(x, y, w=None):
    x_arr = as_float_array(x)
    y_arr = as_float_array(y)
    if len(x_arr) != len(y_arr):
        return max(_lb_keogh(y_arr, x_arr, w), _lb_keogh(x_arr, y_arr, w))
    window = resolve_window(len(x_arr), len(y_arr), w)
    upper_y, lower_y = make_envelopes(y_arr, window)
    projected = np.clip(x_arr, lower_y, upper_y)
    upper_h, lower_h = make_envelopes(projected, window)
    return math.sqrt(
        _lb_keogh_square(x_arr, upper_y, lower_y) + _lb_keogh_square(y_arr, upper_h, lower_h)
    )


def lcss(x, y, epsilon, w=None, fast=True):
    return _dispatch_fast(fast, lcss_scb_numba, lcss_scb, x, y, epsilon, w)


def lcss_n(x, y, epsilon):
    return _lcss_distance_impl(x, y, epsilon)


def lcss_scb(x, y, epsilon, w):
    return _lcss_distance_impl(x, y, epsilon, w)


def lcss_n_numba(x, y, epsilon):
    return _lcss_distance_impl(x, y, epsilon)


def lcss_scb_numba(x, y, epsilon, w):
    return _lcss_distance_impl(x, y, epsilon, w)


def dlcss(x, y, epsilon, constraint=None, w=None):
    mode = _constraint_mode(constraint)
    if mode is None:
        return dlcss_n(x, y, epsilon)
    return dlcss_scb(x, y, epsilon, w)


def dlcss_n(x, y, epsilon):
    return _lcss_distance_impl(_derivative_series(x), _derivative_series(y), epsilon)


def dlcss_scb(x, y, epsilon, w):
    return _lcss_distance_impl(_derivative_series(x), _derivative_series(y), epsilon, w)


def lb_keogh_lcss_n(y, x, epsilon, w):
    return _lb_keogh_lcss_impl(y, x, epsilon, w)


def lb_keogh_lcss_numba(y, x, epsilon, w):
    return _lb_keogh_lcss_impl(y, x, epsilon, w)


def erp(x, y, m, constraint=None, w=None, fast=True):
    mode = _constraint_mode(constraint)
    if mode is None:
        return _dispatch_fast(fast, erp_n_numba, erp_n, x, y, m)
    return _dispatch_fast(fast, erp_scb_numba, erp_scb, x, y, m, w)


def erp_n_numba(x, y, m):
    return _erp_distance_impl(x, y, m)


def erp_n(x, y, m):
    return _erp_distance_impl(x, y, m)


def erp_scb(x, y, m, w):
    return _erp_distance_impl(x, y, m, w)


def erp_scb_numba(x, y, m, w):
    return _erp_distance_impl(x, y, m, w)


def lb_erp(x, y, fast=True):
    return _dispatch_fast(fast, lb_erp_numba, lb_erp_n, x, y)


def lb_erp_n(x, y):
    return abs(np.sum(as_float_array(x)) - np.sum(as_float_array(y)))


def lb_erp_numba(x, y):
    return lb_erp_n(x, y)


def lb_keogh_erp(y, x, m, w=None, fast=True):
    return _dispatch_fast(fast, lb_keogh_erp_numba, lb_keogh_erp_n, y, x, m, w)


def lb_keogh_erp_n(y, x, m, w):
    return _lb_keogh_erp_impl(y, x, m, w)


def lb_keogh_erp_numba(y, x, m, w):
    return _lb_keogh_erp_impl(y, x, m, w)


def lb_kim_erp(y, x, m, fast=True):
    return _dispatch_fast(fast, lb_kim_erp_numba, lb_kim_erp_n, y, x, m)


def lb_kim_erp_n(y, x, m):
    return _lb_kim_erp_impl(y, x, m)


def lb_kim_erp_numba(y, x, m):
    return _lb_kim_erp_impl(y, x, m)


def glb_erp(y, x, m, w):
    x_arr = as_float_array(x)
    y_arr = as_float_array(y)
    window = resolve_window(len(x_arr), len(y_arr), w)
    fixed_dist = min(abs(x_arr[-1] - y_arr[-1]), abs(x_arr[-1] - m), abs(y_arr[-1] - m))
    y_dist = 0.0
    for idx in range(1, len(y_arr) - 1):
        lower, upper = window_minmax(x_arr, idx, window)
        if y_arr[idx] > upper:
            y_dist += min(abs(y_arr[idx] - upper), abs(y_arr[idx] - m))
        elif y_arr[idx] < lower:
            y_dist += min(abs(y_arr[idx] - lower), abs(y_arr[idx] - m))
    x_dist = 0.0
    for idx in range(1, len(x_arr) - 1):
        lower, upper = window_minmax(y_arr, idx, window)
        if x_arr[idx] > upper:
            x_dist += min(abs(x_arr[idx] - upper), abs(x_arr[idx] - m))
        elif x_arr[idx] < lower:
            x_dist += min(abs(x_arr[idx] - lower), abs(x_arr[idx] - m))
    return fixed_dist + max(x_dist, y_dist)


def edr(x, y, m=0, constraint=None, w=100, fast=True):
    mode = _constraint_mode(constraint)
    if mode is None:
        return _dispatch_fast(fast, edr_n_numba, edr_n, x, y, m)
    return _dispatch_fast(fast, edr_scb_numba, edr_scb, x, y, m, w)


def edr_n(x, y, m):
    return _edr_distance_impl(x, y, m)


def edr_n_numba(x, y, m):
    return _edr_distance_impl(x, y, m)


def edr_scb(x, y, m, w):
    return _edr_distance_impl(x, y, m, w)


def edr_scb_numba(x, y, m, w):
    return _edr_distance_impl(x, y, m, w)


def twed(x, timesx, y, timesy, lamb, nu, constraint=None, w=None, fast=True):
    mode = _constraint_mode(constraint)
    if mode is None:
        return _dispatch_fast(fast, twed_n_numba, twed_n, x, timesx, y, timesy, lamb, nu)
    return _dispatch_fast(
        fast, twed_scb_numba, twed_scb, x, timesx, y, timesy, lamb, nu, w
    )


def twed_n_numba(x, timesx, y, timesy, lamb, nu):
    return _twed_distance_impl(x, timesx, y, timesy, lamb, nu)


def twed_n(x, timesx, y, timesy, lamb, nu):
    return _twed_distance_impl(x, timesx, y, timesy, lamb, nu)


def twed_scb_numba(x, timesx, y, timesy, lamb, nu, w=None):
    return _twed_distance_impl(x, timesx, y, timesy, lamb, nu, w)


def twed_scb(x, timesx, y, timesy, lamb, nu, w=None):
    return _twed_distance_impl(x, timesx, y, timesy, lamb, nu, w)


def lb_twed(y, x, lamb, nu, w=None, fast=True):
    return _dispatch_fast(fast, lb_twed_numba, lb_twed_n, y, x, lamb, nu, w)


def lb_twed_n(y, x, lamb, nu, w):
    return _lb_twed_impl(y, x, lamb, nu, w)


def lb_twed_numba(y, x, lamb, nu, w):
    return _lb_twed_impl(y, x, lamb, nu, w)


def glb_twed(x, y, lamb, w):
    return max(_lb_twed_impl(y, x, lamb, 0.0, w), _lb_twed_impl(x, y, lamb, 0.0, w))


def msm(x, y, c, constraint=None, w=None, fast=True):
    mode = _constraint_mode(constraint)
    if mode is None:
        return _dispatch_fast(fast, msm_n_numba, msm_n, x, y, c)
    return _dispatch_fast(fast, msm_scb_numba, msm_scb, x, y, c, w)


def msm_n_numba(x, y, c):
    return _msm_distance_impl(x, y, c)


def msm_n(x, y, c):
    return _msm_distance_impl(x, y, c)


def msm_scb_numba(x, y, c, w):
    return _msm_distance_impl(x, y, c, w)


def msm_scb(x, y, c, w):
    return _msm_distance_impl(x, y, c, w)


def lb_msm_n(y, x, c, w=None):
    return _lb_msm_impl(y, x, c, w)


def lb_msm_numba(y, x, c, w=None):
    return _lb_msm_impl(y, x, c, w)


def swale(x, y, p, r, epsilon, constraint=None, w=None, fast=True):
    mode = _constraint_mode(constraint)
    if mode is None:
        return _dispatch_fast(fast, swale_n_numba, swale_n, x, y, p, r, epsilon)
    return _dispatch_fast(fast, swale_scb_numba, swale_scb, x, y, p, r, epsilon, w)


def swale_n_numba(x, y, p, r, epsilon):
    return _swale_distance_impl(x, y, p, r, epsilon)


def swale_n(x, y, p, r, epsilon):
    return _swale_distance_impl(x, y, p, r, epsilon)


def swale_scb_numba(x, y, p, r, epsilon, w):
    return _swale_distance_impl(x, y, p, r, epsilon, w)


def swale_scb(x, y, p, r, epsilon, w):
    return _swale_distance_impl(x, y, p, r, epsilon, w)
