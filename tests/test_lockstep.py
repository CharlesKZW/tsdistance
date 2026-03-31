import math

import numpy as np

from tsdistance import lockstep


# Source:
# https://rdrr.io/cran/philentropy/src/R/distance.R


def test_corrected_lockstep_formulas_match_philentropy_definitions():
    x = np.array([0.4, 0.6], dtype=float)
    y = np.array([0.5, 0.5], dtype=float)

    assert math.isclose(lockstep.Intersection(x, y), np.sum(np.minimum(x, y)))
    assert math.isclose(lockstep.motyka(x, y), np.sum(np.minimum(x, y)) / np.sum(x + y))
    assert math.isclose(lockstep.tanimoto(x, y), np.sum(np.abs(x - y)) / np.sum(np.maximum(x, y)))
    assert math.isclose(lockstep.harmonicmean(x, y), np.sum((2 * x * y) / (x + y)))

    jaccard = np.sum((x - y) ** 2) / (np.sum(x**2) + np.sum(y**2) - np.sum(x * y))
    assert math.isclose(lockstep.jaccard(x, y), jaccard)

    clark = math.sqrt(np.sum((np.abs(x - y) / (x + y)) ** 2))
    assert math.isclose(lockstep.clark(x, y), clark)

    emanon4 = np.sum(((x - y) ** 2) / (np.maximum(x, y) ** 2))
    assert math.isclose(lockstep.emanon4(x, y), emanon4)


def test_probability_measures_normalize_positive_histograms():
    x = np.array([2.0, 3.0, 5.0])
    y = np.array([4.0, 4.0, 2.0])
    xp = x / x.sum()
    yp = y / y.sum()

    bc = np.sum(np.sqrt(xp * yp))
    hellinger = np.sqrt(np.sum((np.sqrt(xp) - np.sqrt(yp)) ** 2)) / np.sqrt(2.0)
    matusita = np.sqrt(np.sum((np.sqrt(xp) - np.sqrt(yp)) ** 2))

    assert math.isclose(lockstep.bhattacharyya(x, y), -math.log(bc))
    assert math.isclose(lockstep.hellinger(x, y), hellinger)
    assert math.isclose(lockstep.matusita(x, y), matusita)


def test_aliases_resolve_to_canonical_implementations():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([1.0, 3.0, 5.0])
    assert lockstep.ED(x, y) == lockstep.abs_euclidean(x, y)
    assert lockstep.kulczynski(x, y) == lockstep.Kulczynski(x, y)
    assert lockstep.jansen_shannon(np.array([0.4, 0.6]), np.array([0.5, 0.5])) == lockstep.jensen_shannon(
        np.array([0.4, 0.6]), np.array([0.5, 0.5])
    )
    assert lockstep.square_chord(np.array([0.4, 0.6]), np.array([0.5, 0.5])) == lockstep.Square_chord(
        np.array([0.4, 0.6]), np.array([0.5, 0.5])
    )
