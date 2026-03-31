import math

import numpy as np

from tsdistance import elastic


# Source references for the recurrences used below:
# - DDTW: Keogh & Pazzani (2001) and aeon reference source:
#   https://raw.githubusercontent.com/aeon-toolkit/aeon/main/aeon/distances/elastic/_ddtw.py
# - LCSS: Vlachos et al. (2002) and aeon reference source:
#   https://raw.githubusercontent.com/aeon-toolkit/aeon/main/aeon/distances/elastic/_lcss.py
# - ERP / EDR: Chen & Ng (2004), Fig. 2:
#   https://www.vldb.org/conf/2004/RS21P2.PDF
# - TWE: Marteau (2009) and aeon reference source:
#   https://raw.githubusercontent.com/aeon-toolkit/aeon/main/aeon/distances/elastic/_twe.py
# - MSM: Stefan et al. (2013) and aeon reference source:
#   https://raw.githubusercontent.com/aeon-toolkit/aeon/main/aeon/distances/elastic/_msm.py
# - DLCSS: Gorecki (2014):
#   https://www.sciencedirect.com/science/article/pii/S0167865514000890


def reference_dtw_root(x, y, w=None):
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if w is None:
        w = max(len(x_arr), len(y_arr))
    cost = np.full((len(x_arr) + 1, len(y_arr) + 1), np.inf)
    cost[0, 0] = 0.0
    for i in range(1, len(x_arr) + 1):
        for j in range(max(1, i - w), min(len(y_arr), i + w) + 1):
            point = (x_arr[i - 1] - y_arr[j - 1]) ** 2
            cost[i, j] = point + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    return math.sqrt(cost[-1, -1])


def reference_derivative(series):
    arr = np.asarray(series, dtype=float)
    return np.array(
        [
            ((arr[i] - arr[i - 1]) + (arr[i + 1] - arr[i - 1]) / 2.0) / 2.0
            for i in range(1, len(arr) - 1)
        ],
        dtype=float,
    )


def reference_lcss(x, y, epsilon, w=None):
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if w is None:
        w = max(len(x_arr), len(y_arr))
    cost = np.zeros((len(x_arr) + 1, len(y_arr) + 1), dtype=float)
    for i in range(1, len(x_arr) + 1):
        for j in range(max(1, i - w), min(len(y_arr), i + w) + 1):
            if abs(x_arr[i - 1] - y_arr[j - 1]) <= epsilon:
                cost[i, j] = cost[i - 1, j - 1] + 1.0
            else:
                cost[i, j] = max(cost[i - 1, j], cost[i, j - 1])
    return 1.0 - (cost[-1, -1] / min(len(x_arr), len(y_arr)))


def reference_erp(x, y, gap, w=None):
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if w is None:
        w = max(len(x_arr), len(y_arr))
    cost = np.full((len(x_arr) + 1, len(y_arr) + 1), np.inf)
    cost[0, 0] = 0.0
    for i in range(1, len(x_arr) + 1):
        cost[i, 0] = cost[i - 1, 0] + abs(x_arr[i - 1] - gap)
    for j in range(1, len(y_arr) + 1):
        cost[0, j] = cost[0, j - 1] + abs(y_arr[j - 1] - gap)
    for i in range(1, len(x_arr) + 1):
        for j in range(max(1, i - w), min(len(y_arr), i + w) + 1):
            cost[i, j] = min(
                cost[i - 1, j - 1] + abs(x_arr[i - 1] - y_arr[j - 1]),
                cost[i - 1, j] + abs(x_arr[i - 1] - gap),
                cost[i, j - 1] + abs(y_arr[j - 1] - gap),
            )
    return float(cost[-1, -1])


def reference_edr(x, y, epsilon, w=None):
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if w is None:
        w = max(len(x_arr), len(y_arr))
    cost = np.full((len(x_arr) + 1, len(y_arr) + 1), np.inf)
    cost[:, 0] = np.arange(len(x_arr) + 1, dtype=float)
    cost[0, :] = np.arange(len(y_arr) + 1, dtype=float)
    for i in range(1, len(x_arr) + 1):
        for j in range(max(1, i - w), min(len(y_arr), i + w) + 1):
            subcost = 0.0 if abs(x_arr[i - 1] - y_arr[j - 1]) <= epsilon else 1.0
            cost[i, j] = min(
                cost[i - 1, j - 1] + subcost,
                cost[i - 1, j] + 1.0,
                cost[i, j - 1] + 1.0,
            )
    return float(cost[-1, -1])


def reference_twed(x, timesx, y, timesy, lamb, nu, w=None):
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    tx = np.asarray(timesx, dtype=float)
    ty = np.asarray(timesy, dtype=float)
    if w is None:
        w = max(len(x_arr), len(y_arr))
    x_pad = np.concatenate(([0.0], x_arr))
    y_pad = np.concatenate(([0.0], y_arr))
    tx_pad = np.concatenate(([0.0], tx))
    ty_pad = np.concatenate(([0.0], ty))
    cost = np.full((len(x_arr) + 1, len(y_arr) + 1), np.inf)
    cost[0, 0] = 0.0
    for i in range(1, len(x_arr) + 1):
        cost[i, 0] = (
            cost[i - 1, 0]
            + abs(x_pad[i] - x_pad[i - 1])
            + nu * abs(tx_pad[i] - tx_pad[i - 1])
            + lamb
        )
    for j in range(1, len(y_arr) + 1):
        cost[0, j] = (
            cost[0, j - 1]
            + abs(y_pad[j] - y_pad[j - 1])
            + nu * abs(ty_pad[j] - ty_pad[j - 1])
            + lamb
        )
    for i in range(1, len(x_arr) + 1):
        for j in range(max(1, i - w), min(len(y_arr), i + w) + 1):
            cost[i, j] = min(
                cost[i - 1, j]
                + abs(x_pad[i] - x_pad[i - 1])
                + nu * abs(tx_pad[i] - tx_pad[i - 1])
                + lamb,
                cost[i, j - 1]
                + abs(y_pad[j] - y_pad[j - 1])
                + nu * abs(ty_pad[j] - ty_pad[j - 1])
                + lamb,
                cost[i - 1, j - 1]
                + abs(x_pad[i] - y_pad[j])
                + abs(x_pad[i - 1] - y_pad[j - 1])
                + nu * (abs(tx_pad[i] - ty_pad[j]) + abs(tx_pad[i - 1] - ty_pad[j - 1])),
            )
    return float(cost[-1, -1])


def reference_msm_cost(new_point, previous_same_series, other_point, c):
    if (previous_same_series <= new_point <= other_point) or (
        previous_same_series >= new_point >= other_point
    ):
        return c
    return c + min(
        abs(new_point - previous_same_series),
        abs(new_point - other_point),
    )


def reference_msm(x, y, c, w=None):
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if w is None:
        w = max(len(x_arr), len(y_arr))
    cost = np.full((len(x_arr), len(y_arr)), np.inf)
    cost[0, 0] = abs(x_arr[0] - y_arr[0])
    for i in range(1, len(x_arr)):
        cost[i, 0] = cost[i - 1, 0] + reference_msm_cost(x_arr[i], x_arr[i - 1], y_arr[0], c)
    for j in range(1, len(y_arr)):
        cost[0, j] = cost[0, j - 1] + reference_msm_cost(y_arr[j], y_arr[j - 1], x_arr[0], c)
    for i in range(1, len(x_arr)):
        for j in range(max(1, i - w), min(len(y_arr) - 1, i + w) + 1):
            cost[i, j] = min(
                cost[i - 1, j - 1] + abs(x_arr[i] - y_arr[j]),
                cost[i - 1, j] + reference_msm_cost(x_arr[i], x_arr[i - 1], y_arr[j], c),
                cost[i, j - 1] + reference_msm_cost(y_arr[j], y_arr[j - 1], x_arr[i], c),
            )
    return float(cost[-1, - 1])


def reference_swale(x, y, p, r, epsilon, w=None):
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if w is None:
        w = max(len(x_arr), len(y_arr))
    cost = np.full((len(x_arr) + 1, len(y_arr) + 1), np.inf)
    cost[:, 0] = np.arange(len(x_arr) + 1, dtype=float) * p
    cost[0, :] = np.arange(len(y_arr) + 1, dtype=float) * p
    for i in range(1, len(x_arr) + 1):
        for j in range(max(1, i - w), min(len(y_arr), i + w) + 1):
            if abs(x_arr[i - 1] - y_arr[j - 1]) <= epsilon:
                cost[i, j] = cost[i - 1, j - 1] + r
            else:
                cost[i, j] = min(cost[i - 1, j], cost[i, j - 1]) + p
    return float(cost[-1, -1])


def test_default_constraint_dispatchers_work_with_python_none():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([1.5, 1.9, 3.1, 3.8, 5.2])
    tx = np.arange(len(x), dtype=float)
    ty = np.arange(len(y), dtype=float)

    assert elastic.ddtw(x, y) == elastic.ddtw(x, y, constraint="None")
    assert elastic.dlcss(x, y, 0.5) == elastic.dlcss(x, y, 0.5, constraint="None")
    assert elastic.erp(x, y, 0.0) == elastic.erp(x, y, 0.0, constraint="None")
    assert elastic.edr(x, y, 0.5) == elastic.edr(x, y, 0.5, constraint="None")
    assert elastic.twed(x, tx, y, ty, 1.0, 0.1) == elastic.twed(
        x, tx, y, ty, 1.0, 0.1, constraint="None"
    )
    assert elastic.msm(x, y, 1.0) == elastic.msm(x, y, 1.0, constraint="None")
    assert elastic.swale(x, y, 1.0, 1.0, 0.5) == elastic.swale(
        x, y, 1.0, 1.0, 0.5, constraint="None"
    )


def test_ddtw_matches_reference_recurrence():
    x = np.array([3, 2, 4, 5, 5, 2, 4, 7, 9, 8], dtype=float)
    y = np.array([3, 3, 3, 1, 6, 9, 9], dtype=float)
    expected = reference_dtw_root(reference_derivative(x), reference_derivative(y), w=3)
    assert math.isclose(elastic.ddtw(x, y, constraint="Sakoe-Chiba", w=3), expected)


def test_lcss_and_dlcss_match_reference_recurrences():
    x = np.array([3, 4, 38, 4, 5], dtype=float)
    y = np.array([0, 3, 4], dtype=float)
    epsilon = 0.7
    assert math.isclose(elastic.lcss(x, y, epsilon, fast=False), reference_lcss(x, y, epsilon))
    assert math.isclose(
        elastic.dlcss(x, y, epsilon, constraint="Sakoe-Chiba", w=2),
        reference_lcss(reference_derivative(x), reference_derivative(y), epsilon, w=2),
    )


def test_erp_matches_paper_recurrence_and_bounds_hold():
    x = np.array([3, 4, 38, 4, 5], dtype=float)
    y = np.array([0, 3, 4], dtype=float)
    expected = reference_erp(x, y, gap=0.0)
    assert math.isclose(elastic.erp(x, y, m=0.0, fast=False), expected)

    rng = np.random.default_rng(0)
    for _ in range(25):
        a = rng.normal(size=8)
        b = rng.normal(size=8)
        dist = elastic.erp(a, b, m=0.0, constraint="Sakoe-Chiba", w=3, fast=False)
        assert elastic.lb_erp(a, b, fast=False) <= dist + 1e-9
        assert elastic.lb_keogh_erp(b, a, m=0.0, w=3, fast=False) <= dist + 1e-9
        assert elastic.lb_kim_erp(b, a, m=0.0, fast=False) <= elastic.erp(a, b, m=0.0) + 1e-9
        assert elastic.glb_erp(b, a, m=0.0, w=3) <= dist + 1e-9


def test_edr_matches_paper_recurrence():
    x = np.array([3, 4, 74, 4, 5], dtype=float)
    y = np.array([0, 3, 4], dtype=float)
    epsilon = 4.0
    assert math.isclose(elastic.edr(x, y, epsilon, fast=False), reference_edr(x, y, epsilon))


def test_twed_matches_reference_recurrence_and_bounds_hold():
    x = np.array([3, 4, 76, 4, 5], dtype=float)
    y = np.array([0, 3, 4], dtype=float)
    tx = np.arange(len(x), dtype=float)
    ty = np.arange(len(y), dtype=float)
    expected = reference_twed(x, tx, y, ty, lamb=2.5, nu=1.0, w=5)
    assert math.isclose(
        elastic.twed(x, tx, y, ty, lamb=2.5, nu=1.0, constraint="Sakoe-Chiba", w=5, fast=False),
        expected,
    )

    rng = np.random.default_rng(1)
    for _ in range(25):
        a = rng.normal(size=8)
        b = rng.normal(size=8)
        ta = np.arange(len(a), dtype=float)
        tb = np.arange(len(b), dtype=float)
        dist = elastic.twed(a, ta, b, tb, 1.0, 0.1, constraint="Sakoe-Chiba", w=3, fast=False)
        assert elastic.lb_twed(b, a, lamb=1.0, nu=0.1, w=3, fast=False) <= dist + 1e-9
        assert elastic.glb_twed(a, b, lamb=1.0, w=3) <= dist + 1e-9


def test_msm_matches_reference_recurrence_and_bound_holds():
    x = np.array([3, 4, 76, 4, 5], dtype=float)
    y = np.array([0, 3, 4], dtype=float)
    expected = reference_msm(x, y, c=1.0, w=5)
    assert math.isclose(elastic.msm(x, y, c=1.0, constraint="Sakoe-Chiba", w=5, fast=False), expected)

    rng = np.random.default_rng(2)
    for _ in range(25):
        a = rng.normal(size=8)
        b = rng.normal(size=8)
        dist = elastic.msm(a, b, c=1.0, constraint="Sakoe-Chiba", w=3, fast=False)
        assert elastic.lb_msm(b, a, c=1.0, w=3, fast=False) <= dist + 1e-9


def test_swale_matches_reference_recurrence():
    x = np.array([3, 4, 76, 4, 5], dtype=float)
    y = np.array([0, 3, 4], dtype=float)
    expected = reference_swale(x, y, p=1.0, r=1.0, epsilon=3.0)
    assert math.isclose(elastic.swale(x, y, p=1.0, r=1.0, epsilon=3.0, fast=False), expected)


def test_lb_improved_handles_unequal_lengths_without_crashing():
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y = np.array([0.0, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5])
    value = elastic.lb_improved(x, y, w=2, fast=False)
    assert value >= 0.0
    assert value <= elastic.dtw(x, y, w=2, fast=False) + 1e-9
